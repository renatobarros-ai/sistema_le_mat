# utils/evaluation_metrics.py
# Módulo para cálculo de métricas de avaliação do modelo
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Implementação robusta de métricas ROUGE e BLEU para avaliação de modelos de texto

import numpy as np
from evaluate import load
import logging
import torch

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred, tokenizer):
    """
    Calcula métricas de avaliação ROUGE e BLEU com tratamento robusto de dados
    
    Esta função implementa um sistema robusto para calcular métricas de avaliação
    de modelos de linguagem, lidando com diferentes formatos de entrada e
    estruturas de dados complexas que podem surgir durante o treinamento.
    
    Args:
        eval_pred: Estrutura contendo predições e labels do modelo
        tokenizer: Tokenizador para decodificar as sequências
        
    Returns:
        dict: Dicionário com métricas calculadas (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BLEU)
    """
    try:
        # Extração segura das predições e labels
        predictions, labels = _extract_predictions_and_labels(eval_pred)
        
        # Conversão para formato adequado
        predictions, labels = _convert_to_numpy(predictions, labels)
        
        # Tratamento de estruturas complexas
        predictions = _handle_complex_structures(predictions)
        labels = _handle_complex_structures(labels)
        
        # Verificação e ajuste de dimensões
        predictions = _adjust_dimensions(predictions)
        labels = _adjust_dimensions(labels)
        
        logger.info(f"Processando {predictions.shape[0]} exemplos para métricas")
        
        # Decodificação das sequências
        decoded_preds, decoded_labels = _decode_sequences(predictions, labels, tokenizer)
        
        # Verificação de dados suficientes
        if len(decoded_preds) == 0:
            logger.warning("Nenhum par válido de predição/label foi decodificado")
            return _get_zero_metrics()
        
        # Log de amostra para debugging
        _log_sample_data(decoded_preds, decoded_labels)
        
        # Cálculo das métricas
        return _calculate_metrics(decoded_preds, decoded_labels)
        
    except Exception as e:
        logger.error(f"Erro no cálculo de métricas: {str(e)}", exc_info=True)
        return _get_zero_metrics_with_error(str(e))

def _extract_predictions_and_labels(eval_pred):
    """
    Extrai predições e labels de diferentes formatos de entrada
    
    Args:
        eval_pred: Estrutura de dados com predições e labels
        
    Returns:
        tuple: (predictions, labels)
    """
    if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
        return eval_pred.predictions, eval_pred.label_ids
    elif isinstance(eval_pred, tuple) and len(eval_pred) == 2:
        return eval_pred
    else:
        raise ValueError(f"Formato de eval_pred não reconhecido: {type(eval_pred)}")

def _convert_to_numpy(predictions, labels):
    """
    Converte tensors para numpy arrays se necessário
    
    Args:
        predictions: Predições do modelo
        labels: Labels verdadeiros
        
    Returns:
        tuple: (predictions, labels) como numpy arrays
    """
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    return predictions, labels

def _handle_complex_structures(data):
    """
    Trata estruturas de dados complexas e irregulares
    
    Args:
        data: Dados a serem processados
        
    Returns:
        np.ndarray: Dados processados
    """
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and hasattr(data[0], 'shape'):
            # Primeira dimensão relevante
            data = data[0]
        elif len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
            # Converter lista de listas para array
            try:
                data = np.array(data[0])
            except:
                logger.warning("Não foi possível converter dados para array")
                return np.array([])
    
    return data

def _adjust_dimensions(data):
    """
    Ajusta dimensões dos dados para formato adequado
    
    Args:
        data: Dados a serem ajustados
        
    Returns:
        np.ndarray: Dados com dimensões ajustadas
    """
    if not hasattr(data, 'shape'):
        return data
    
    # Se tem 3 dimensões (batch, seq_len, vocab_size), aplicar argmax
    if len(data.shape) == 3:
        data = np.argmax(data, axis=-1)
    # Se tem mais de 3 dimensões, reformatar
    elif len(data.shape) > 3:
        data = data.reshape(-1, data.shape[-1])
        if data.shape[1] > 1:  # Se ainda tem dimensão do vocabulário
            data = np.argmax(data, axis=-1)
    
    # Garantir que é 2D (batch_size, seq_len)
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    return data

def _decode_sequences(predictions, labels, tokenizer):
    """
    Decodifica sequências de tokens para texto
    
    Args:
        predictions: Predições tokenizadas
        labels: Labels tokenizados
        tokenizer: Tokenizador para decodificação
        
    Returns:
        tuple: (decoded_predictions, decoded_labels)
    """
    # Conversão para formato adequado
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Limpeza de labels (remover -100)
    labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = []
    decoded_labels = []
    
    # Processamento exemplo por exemplo
    for i in range(min(len(predictions), len(labels_clean))):
        try:
            # Decodificação da predição
            pred_text = _decode_single_sequence(predictions[i], tokenizer)
            
            # Decodificação do label
            label_text = _decode_single_sequence(labels_clean[i], tokenizer)
            
            # Adicionar apenas se ambos são válidos
            if pred_text and label_text:
                decoded_preds.append(pred_text)
                decoded_labels.append(label_text)
                
        except Exception as e:
            logger.debug(f"Erro ao decodificar exemplo {i}: {str(e)}")
            continue
    
    return decoded_preds, decoded_labels

def _decode_single_sequence(tokens, tokenizer):
    """
    Decodifica uma única sequência de tokens
    
    Args:
        tokens: Sequência de tokens
        tokenizer: Tokenizador
        
    Returns:
        str: Texto decodificado
    """
    if isinstance(tokens, (list, np.ndarray)):
        # Filtrar tokens válidos
        valid_tokens = [int(t) for t in tokens 
                       if isinstance(t, (int, np.integer)) and 0 <= t < tokenizer.vocab_size]
        if len(valid_tokens) > 0:
            return tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
    
    return ""

def _log_sample_data(decoded_preds, decoded_labels):
    """
    Registra dados de amostra para debugging
    
    Args:
        decoded_preds: Predições decodificadas
        decoded_labels: Labels decodificados
    """
    if len(decoded_preds) > 0:
        logger.info(f"Successfully decoded {len(decoded_preds)} pairs")
        sample_pred = decoded_preds[0][:100] + ('...' if len(decoded_preds[0]) > 100 else '')
        sample_label = decoded_labels[0][:100] + ('...' if len(decoded_labels[0]) > 100 else '')
        logger.info(f"Amostra predição: '{sample_pred}'")
        logger.info(f"Amostra label: '{sample_label}'")

def _calculate_metrics(decoded_preds, decoded_labels):
    """
    Calcula as métricas ROUGE e BLEU
    
    Args:
        decoded_preds: Predições decodificadas
        decoded_labels: Labels decodificados
        
    Returns:
        dict: Dicionário com métricas calculadas
    """
    # Carregamento das métricas
    rouge = load("rouge")
    bleu = load("bleu")
    
    # Cálculo ROUGE
    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Cálculo BLEU
    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    
    # Compilação dos resultados
    result = {
        'rouge1': round(rouge_result['rouge1'], 4),
        'rouge2': round(rouge_result['rouge2'], 4),
        'rougeL': round(rouge_result['rougeL'], 4),
        'rougeLsum': round(rouge_result['rougeLsum'], 4),
        'bleu': round(bleu_result['bleu'], 4)
    }
    
    # Log dos resultados
    logger.info("✅ Métricas calculadas com sucesso!")
    logger.info(f"📊 ROUGE-L: {result['rougeL']:.4f}, BLEU: {result['bleu']:.4f}")
    logger.info(f"📈 Pares processados: {len(decoded_preds)}")
    
    return result

def _get_zero_metrics():
    """
    Retorna métricas zeradas para casos de erro
    
    Returns:
        dict: Dicionário com métricas zeradas
    """
    return {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0,
        'bleu': 0.0
    }

def _get_zero_metrics_with_error(error_message):
    """
    Retorna métricas zeradas com mensagem de erro
    
    Args:
        error_message (str): Mensagem de erro
        
    Returns:
        dict: Dicionário com métricas zeradas e erro
    """
    return {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0,
        'bleu': 0.0,
        'error': error_message[:100]  # Limitar tamanho da mensagem
    }