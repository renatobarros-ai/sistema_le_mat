# utils/evaluation_metrics.py - VERSÃO FINAL CORRIGIDA
import numpy as np
from evaluate import load
import logging
import torch

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred, tokenizer):
    """
    Calcula métricas de avaliação ROUGE e BLEU com tratamento ultra-robusto.
    VERSÃO FINAL: Trata estruturas complexas e irregulares das predictions.
    """
    try:
        # Extração segura das predições e labels
        if hasattr(eval_pred, 'predictions') and hasattr(eval_pred, 'label_ids'):
            predictions, labels = eval_pred.predictions, eval_pred.label_ids
        elif isinstance(eval_pred, tuple) and len(eval_pred) == 2:
            predictions, labels = eval_pred
        else:
            logger.error(f"Formato de eval_pred não reconhecido: {type(eval_pred)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 'bleu': 0.0}

        logger.debug(f"Predictions type: {type(predictions)}")
        logger.debug(f"Labels type: {type(labels)}")

        # CORREÇÃO CRÍTICA: Tratamento de estruturas complexas
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Tratamento especial para estruturas irregulares
        if isinstance(predictions, (list, tuple)):
            # Se é lista/tupla, tentar extrair a primeira dimensão relevante
            if len(predictions) > 0 and hasattr(predictions[0], 'shape'):
                # Pegar o primeiro elemento se é uma sequência de arrays
                predictions = predictions[0]
            elif len(predictions) > 0 and isinstance(predictions[0], (list, np.ndarray)):
                # Converter lista de listas para array
                try:
                    predictions = np.array(predictions[0])
                except:
                    logger.warning("Não foi possível converter predictions para array")
                    return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 'bleu': 0.0}

        # Verificar e ajustar dimensões
        if hasattr(predictions, 'shape'):
            logger.debug(f"Predictions shape: {predictions.shape}")
            
            # Se tem 3 dimensões (batch, seq_len, vocab_size), aplicar argmax
            if len(predictions.shape) == 3:
                predictions = np.argmax(predictions, axis=-1)
            # Se tem mais de 3 dimensões, tentar extrair as dimensões relevantes
            elif len(predictions.shape) > 3:
                # Tentar pegar as últimas 2 dimensões (batch_size, seq_len)
                predictions = predictions.reshape(-1, predictions.shape[-1])
                if predictions.shape[1] > 1:  # Se ainda tem vocab dimension
                    predictions = np.argmax(predictions, axis=-1)

        # Garantir que predictions é 2D (batch_size, seq_len)
        if hasattr(predictions, 'shape') and len(predictions.shape) > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)

        # Mesmo tratamento para labels
        if hasattr(labels, 'shape'):
            logger.debug(f"Labels shape: {labels.shape}")
            if len(labels.shape) > 2:
                labels = labels.reshape(labels.shape[0], -1)

        # Converter para formato adequado para decodificação
        try:
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
        except Exception as e:
            logger.error(f"Erro ao converter para numpy: {str(e)}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 'bleu': 0.0}

        logger.info(f"Processando {predictions.shape[0]} exemplos para métricas")

        # Decodificação segura - exemplo por exemplo
        decoded_preds = []
        decoded_labels = []
        
        # Processar labels primeiro (remover -100)
        labels_clean = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        successful_pairs = 0
        for i in range(min(len(predictions), len(labels_clean))):
            try:
                # Decodificar predição
                pred_tokens = predictions[i]
                if isinstance(pred_tokens, (list, np.ndarray)):
                    # Garantir que são inteiros válidos
                    pred_tokens = [int(t) for t in pred_tokens if isinstance(t, (int, np.integer)) and 0 <= t < tokenizer.vocab_size]
                    if len(pred_tokens) > 0:
                        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
                    else:
                        pred_text = ""
                else:
                    pred_text = ""

                # Decodificar label
                label_tokens = labels_clean[i]
                if isinstance(label_tokens, (list, np.ndarray)):
                    label_tokens = [int(t) for t in label_tokens if isinstance(t, (int, np.integer)) and 0 <= t < tokenizer.vocab_size]
                    if len(label_tokens) > 0:
                        label_text = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
                    else:
                        label_text = ""
                else:
                    label_text = ""

                # Só adicionar se ambos não estão vazios
                if pred_text and label_text:
                    decoded_preds.append(pred_text)
                    decoded_labels.append(label_text)
                    successful_pairs += 1

            except Exception as e:
                logger.debug(f"Erro ao decodificar exemplo {i}: {str(e)}")
                continue

        logger.info(f"Successfully decoded {successful_pairs} pairs")

        # Verificar se temos dados suficientes
        if successful_pairs == 0:
            logger.warning("Nenhum par válido de predição/label foi decodificado")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0, 'bleu': 0.0}

        # Log de amostra para debugging
        if len(decoded_preds) > 0:
            logger.info(f"Amostra predição: '{decoded_preds[0][:100]}{'...' if len(decoded_preds[0]) > 100 else ''}'")
            logger.info(f"Amostra label: '{decoded_labels[0][:100]}{'...' if len(decoded_labels[0]) > 100 else ''}'")

        # Calcular métricas
        rouge = load("rouge")
        bleu = load("bleu")

        rouge_result = rouge.compute(
            predictions=decoded_preds, 
            references=decoded_labels,
            use_stemmer=True
        )
        
        bleu_result = bleu.compute(
            predictions=decoded_preds, 
            references=[[label] for label in decoded_labels]
        )

        result = {
            'rouge1': round(rouge_result['rouge1'], 4),
            'rouge2': round(rouge_result['rouge2'], 4),
            'rougeL': round(rouge_result['rougeL'], 4),
            'rougeLsum': round(rouge_result['rougeLsum'], 4),
            'bleu': round(bleu_result['bleu'], 4)
        }
        
        logger.info(f"✅ Métricas calculadas com sucesso!")
        logger.info(f"📊 ROUGE-L: {result['rougeL']:.4f}, BLEU: {result['bleu']:.4f}")
        logger.info(f"📈 Pares processados: {successful_pairs}/{len(predictions)}")
        
        return result
        
    except Exception as e:
        logger.error(f"ERRO CRÍTICO no cálculo de métricas: {str(e)}", exc_info=True)
        return {
            'rouge1': 0.0, 
            'rouge2': 0.0, 
            'rougeL': 0.0, 
            'rougeLsum': 0.0, 
            'bleu': 0.0,
            'error': str(e)[:100]  # Limitar tamanho do erro
        }
