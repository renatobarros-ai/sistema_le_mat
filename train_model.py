# train_model.py - Sistema de Fine-tuning PTT5 com QLoRA - VERSÃO OTIMIZADA
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Script principal OTIMIZADO para fine-tuning baseado em análise de resultados

import pandas as pd
import json
import logging
import os
import signal
import sys
import warnings
from datetime import datetime
import yaml

from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# Importações dos módulos customizados
from prompts.pessoa_x_prompt import PESSOA_X_FINE_TUNING_PROMPT
from utils.data_processing import prepare_data_for_fine_tuning, tokenize_function, validate_dataframe
from utils.evaluation_metrics import compute_metrics

# Configuração inicial do ambiente
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# Configuração do sistema de logging OTIMIZADO
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"./results/training_log_optimized_{timestamp}.log"
os.makedirs("./results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(sig, frame):
    """
    Manipula interrupções do usuário (Ctrl+C) de forma segura
    """
    logger.info("⚠️ Treinamento interrompido pelo usuário (Ctrl+C)")
    logger.info(f"📝 Log salvo em: {log_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def log_model_info(model):
    """
    Log informações detalhadas do modelo para debugging
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Parâmetros totais: {total_params:,}")
    logger.info(f"📊 Parâmetros treináveis: {trainable_params:,}")
    logger.info(f"📊 Percentual treinável: {100 * trainable_params / total_params:.2f}%")

def log_sample_predictions(decoded_preds, decoded_labels, num_samples=3):
    """
    Log amostras de predições para monitoramento de qualidade
    """
    logger.info("🔍 === AMOSTRAS DE PREDIÇÕES ===")
    for i in range(min(num_samples, len(decoded_preds))):
        pred_preview = decoded_preds[i][:150] + ('...' if len(decoded_preds[i]) > 150 else '')
        label_preview = decoded_labels[i][:150] + ('...' if len(decoded_labels[i]) > 150 else '')
        
        logger.info(f"📝 Amostra {i+1}:")
        logger.info(f"   Predição: '{pred_preview}'")
        logger.info(f"   Esperado:  '{label_preview}'")
        logger.info("   " + "="*50)

# Início do processo de fine-tuning OTIMIZADO
logger.info("=== INICIANDO FINE-TUNING PTT5 OTIMIZADO - SISTEMA DE CARTAS ===")
logger.info("🚀 Versão otimizada baseada em análise de resultados anteriores")

try:
    # Etapa 1: Carregamento das configurações
    logger.info("1. Carregando configurações otimizadas de treinamento...")
    with open('config/training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Extração das configurações específicas
    model_config = config['model_config']
    lora_config_data = config['lora_config']
    training_args_data = config['training_args']
    early_stopping_data = config['early_stopping']
    dataset_path = config['dataset_path']
    test_size = config['test_size']
    random_state = config['random_state']

    logger.info(f"✅ Configurações carregadas para o modelo: {model_config['name']}")
    logger.info(f"🔧 Max length otimizado: {model_config['max_length']} (era 450)")
    logger.info(f"🔧 Learning rate otimizado: {training_args_data['learning_rate']} (era 0.00002)")
    logger.info(f"🔧 LoRA rank otimizado: {lora_config_data['r']} (era 16)")

    # Etapa 2: Carregamento e preprocessamento dos dados
    logger.info("2. Carregando e processando dataset...")
    df = pd.read_excel(dataset_path)
    logger.info(f"📊 Dados carregados: {len(df)} registros")
    
    # Validação e limpeza dos dados
    df = validate_dataframe(df, ['texto', 'carta', 'evento', 'secao', 'tema'])
    logger.info(f"✅ Dados após validação/limpeza: {len(df)} registros válidos")

    # Estatísticas dos dados OTIMIZADAS
    logger.info(f"📈 Cartas únicas: {df['carta'].nunique()}")
    logger.info(f"📈 Comprimento médio: {df['texto'].str.len().mean():.1f} caracteres")
    logger.info(f"📈 Comprimento máximo: {df['texto'].str.len().max()} caracteres")
    logger.info(f"📈 Textos > 550 tokens: {(df['texto'].str.len() > 550*4).sum()} ({(df['texto'].str.len() > 550*4).mean()*100:.1f}%)")

    # Etapa 3: Divisão estratificada dos dados
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['carta'])
    logger.info(f"✅ Dataset dividido: {len(train_df)} treino, {len(val_df)} validação")

    # Etapa 4: Preparação dos dados com NOVO TEMPLATE
    logger.info("3. Preparando dados com template OTIMIZADO...")
    logger.info(f"🔧 Novo template: '{PESSOA_X_FINE_TUNING_PROMPT}'")
    
    train_data = prepare_data_for_fine_tuning(train_df, PESSOA_X_FINE_TUNING_PROMPT)
    val_data = prepare_data_for_fine_tuning(val_df, PESSOA_X_FINE_TUNING_PROMPT)

    # Log de amostra do novo formato
    if len(train_data) > 0:
        sample_input = train_data[0]['input'][:100] + ('...' if len(train_data[0]['input']) > 100 else '')
        sample_output = train_data[0]['output'][:100] + ('...' if len(train_data[0]['output']) > 100 else '')
        logger.info(f"📝 Amostra novo formato INPUT: '{sample_input}'")
        logger.info(f"📝 Amostra novo formato OUTPUT: '{sample_output}'")

    # Criação dos datasets do Hugging Face
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    logger.info(f"✅ Datasets criados: {len(train_dataset)} treino, {len(val_dataset)} validação")

    # Etapa 5: Configuração do modelo e tokenizador
    logger.info("4. Carregando modelo base com configurações otimizadas...")
    model_name = model_config['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuração da quantização 4-bit (mantida)
    bnb_compute_dtype = getattr(torch, model_config['quantization']['bnb_4bit_compute_dtype'].split('.')[-1])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config['quantization']['load_in_4bit'],
        bnb_4bit_use_double_quant=model_config['quantization']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=model_config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=bnb_compute_dtype
    )

    # Carregamento do modelo com quantização
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=bnb_compute_dtype
    )
    logger.info(f"✅ Modelo carregado: {model_name} com quantização 4-bit")
    
    # Preparação do modelo para treinamento com quantização
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Etapa 6: Configuração dos adaptadores LoRA OTIMIZADOS
    logger.info("5. Configurando LoRA com parâmetros otimizados...")
    lora_config = LoraConfig(**lora_config_data)
    model = get_peft_model(model, lora_config)
    logger.info("✅ Parâmetros LoRA otimizados configurados")
    model.print_trainable_parameters()
    
    # Log informações detalhadas do modelo
    log_model_info(model)

    # Etapa 7: Tokenização dos dados com novo max_length
    logger.info("6. Tokenizando dados com max_length otimizado...")
    tokenized_train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, model_config['max_length']), 
        batched=True, 
        remove_columns=["input", "output"]
    )
    tokenized_val_dataset = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, model_config['max_length']), 
        batched=True, 
        remove_columns=["input", "output"]
    )
    logger.info(f"✅ Tokenização concluída com max_length {model_config['max_length']}: {len(tokenized_train_dataset)} treino, {len(tokenized_val_dataset)} validação")

    # Configuração do collator de dados
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=model_config['max_length']
    )

    # Etapa 8: Configuração das métricas de avaliação
    logger.info("7. Configurando métricas de avaliação...")
    def curried_compute_metrics(eval_pred):
        """
        Função wrapper para compute_metrics com tokenizer fixo
        """
        return compute_metrics(eval_pred, tokenizer)

    # Etapa 9: Configuração dos argumentos de treinamento OTIMIZADOS
    logger.info("8. Configurando argumentos de treinamento otimizados...")
    training_args = TrainingArguments(**training_args_data)
    
    effective_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    logger.info(f"🔧 Batch efetivo otimizado: {effective_batch} (era 8)")
    logger.info(f"🔧 Total de épocas otimizado: {training_args.num_train_epochs} (era 20)")

    # Etapa 10: Inicialização do Trainer com early stopping otimizado
    logger.info("9. Inicializando Trainer com early stopping otimizado...")
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_data['patience'], 
        early_stopping_threshold=early_stopping_data['threshold']
    )
    
    logger.info(f"🔧 Early stopping: patience={early_stopping_data['patience']} (era 4), threshold={early_stopping_data['threshold']} (era 0.001)")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=curried_compute_metrics,
        callbacks=[early_stopping],
    )
    logger.info("✅ Trainer configurado com parâmetros otimizados")

    # Etapa 11: Execução do treinamento OTIMIZADO
    logger.info("10. INICIANDO TREINAMENTO OTIMIZADO...")
    if torch.cuda.is_available():
        logger.info(f"🖥️ VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.cuda.empty_cache()

    logger.info("🚀 Iniciando com configurações otimizadas baseadas em análise anterior...")
    
    # Execução do treinamento
    training_result = trainer.train()
    logger.info("=== TREINAMENTO OTIMIZADO CONCLUÍDO ===")

    # Etapa 12: Avaliação final do modelo
    logger.info("11. Executando avaliação final...")
    logger.info("⏳ Calculando métricas ROUGE e BLEU otimizadas... (pode demorar 2-3 minutos)")
    final_metrics = trainer.evaluate()
    logger.info("✅ Avaliação final concluída!")

    # Etapa 13: Compilação do relatório de treinamento OTIMIZADO
    logger.info("12. Compilando relatório de treinamento otimizado...")
    training_metrics = {
        'train_runtime': training_result.metrics.get('train_runtime'),
        'train_samples_per_second': training_result.metrics.get('train_samples_per_second'),
        'total_flos': training_result.metrics.get('total_flos', 0),
        'train_loss': training_result.metrics.get('train_loss')
    }

    # Estrutura do relatório final OTIMIZADO
    final_report = {
        'timestamp': timestamp,
        'version': 'optimized',
        'optimizations_applied': [
            'Template redesigned (direct format)',
            'Learning rate increased 5x (0.00002 -> 0.0001)',
            'LoRA rank doubled (16 -> 32)',
            'LoRA alpha doubled (32 -> 64)',
            'Max length increased (450 -> 550)',
            'Early stopping improved (patience 4->2, threshold 0.001->0.01)',
            'Epochs reduced (20 -> 12)',
            'Weight decay reduced (0.01 -> 0.005)'
        ],
        'hardware_info': {
            'gpu': f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'}", 
            'vram': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "N/A",
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else "N/A"
        },
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'unique_cartas': df['carta'].nunique(),
            'avg_text_length': float(df['texto'].str.len().mean()),
            'max_text_length': int(df['texto'].str.len().max()),
            'texts_over_550_tokens': int((df['texto'].str.len() > 550*4).sum())
        },
        'model_config': {
            'model_name': model_name,
            'lora_r': lora_config.r,
            'lora_alpha': lora_config.lora_alpha,
            'lora_dropout': lora_config.lora_dropout,
            'max_length': model_config['max_length'],
            'learning_rate': training_args.learning_rate,
            'effective_batch_size': effective_batch,
            'epochs_completed': training_args.num_train_epochs
        },
        'training_metrics': training_metrics,
        'final_eval_metrics': final_metrics
    }

    # Salvamento do relatório
    logger.info("⏳ Salvando relatório otimizado em JSON...")
    metrics_path = f"./results/training_metrics_optimized_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.info("✅ Relatório otimizado salvo!")
    
    # Exibição dos resultados finais OTIMIZADOS
    logger.info("=== RESULTADOS FINAIS OTIMIZADOS ===")
    logger.info(f"📊 ROUGE-L Final: {final_metrics.get('eval_rougeL', 0.0):.4f}")
    logger.info(f"📊 BLEU Final: {final_metrics.get('eval_bleu', 0.0):.4f}")
    logger.info(f"📊 Loss Final: {final_metrics.get('eval_loss', 0.0):.4f}")
    
    # Comparação com resultados anteriores
    previous_rouge = 0.1398
    previous_bleu = 0.012
    improvement_rouge = ((final_metrics.get('eval_rougeL', 0.0) - previous_rouge) / previous_rouge) * 100
    improvement_bleu = ((final_metrics.get('eval_bleu', 0.0) - previous_bleu) / previous_bleu) * 100
    
    logger.info("🔍 === COMPARAÇÃO COM VERSÃO ANTERIOR ===")
    logger.info(f"📈 Melhoria ROUGE-L: {improvement_rouge:+.1f}%")
    logger.info(f"📈 Melhoria BLEU: {improvement_bleu:+.1f}%")
    logger.info(f"📁 Métricas detalhadas salvas em: {metrics_path}")

    # Etapa 14: Salvamento do modelo treinado OTIMIZADO
    logger.info("13. Salvando modelo otimizado...")
    logger.info("⏳ Salvando adaptadores LoRA otimizados... (pode demorar 1-2 minutos)")
    save_path = f"./model_save/lora_model_462_optimized_{timestamp}"
    trainer.model.save_pretrained(save_path)

    logger.info("⏳ Salvando tokenizer...")
    tokenizer.save_pretrained(save_path)

    logger.info(f"✅ Modelo otimizado salvo com sucesso em: {save_path}")
    logger.info("🎉 === FINE-TUNING OTIMIZADO FINALIZADO COM SUCESSO === 🎉")

except KeyboardInterrupt:
    logger.info("⚠️ Treinamento interrompido pelo usuário (Ctrl+C)")
    logger.info("💾 Salvando checkpoint de emergência...")
    try:
        emergency_path = f"./model_save/emergency_checkpoint_optimized_{timestamp}"
        if 'trainer' in locals():
            trainer.model.save_pretrained(emergency_path)
            logger.info(f"✅ Checkpoint de emergência salvo em: {emergency_path}")
    except Exception as save_error:
        logger.error(f"❌ Erro ao salvar checkpoint de emergência: {str(save_error)}")
    
except FileNotFoundError as e:
    logger.error(f"❌ Arquivo não encontrado: {str(e)}")
    logger.error("🔍 Verifique se os caminhos dos arquivos estão corretos")
    
except yaml.YAMLError as e:
    logger.error(f"❌ Erro ao ler arquivo YAML: {str(e)}")
    logger.error("🔍 Verifique a sintaxe do arquivo config/training_config.yaml")
    
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"❌ Erro de memória GPU: {str(e)}")
    logger.error("💡 Reduza o max_length ou gradient_accumulation_steps no config YAML")
    logger.error("💡 Configuração atual: max_length=550, batch_effective=6")
    
except Exception as e:
    logger.error(f"❌ ERRO CRÍTICO: {str(e)}", exc_info=True)
    logger.error("🔍 Verifique os logs acima para mais detalhes do erro")
    
finally:
    # Limpeza final
    logger.info(f"📝 Log completo salvo em: {log_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cache da GPU limpo")
