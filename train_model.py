# train_model.py - Sistema de Fine-tuning PTT5 com QLoRA
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Script principal para fine-tuning do modelo PTT5 usando quantização 4-bit e adaptadores LoRA

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

# Configuração do sistema de logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"./results/training_log_{timestamp}.log"
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
    logger.info("Treinamento interrompido pelo usuário (Ctrl+C)")
    logger.info(f"Log salvo em: {log_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Início do processo de fine-tuning
logger.info("=== INICIANDO FINE-TUNING PTT5 - SISTEMA DE CARTAS ===")

try:
    # Etapa 1: Carregamento das configurações
    logger.info("1. Carregando configurações de treinamento...")
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

    logger.info(f"Configurações carregadas para o modelo: {model_config['name']}")

    # Etapa 2: Carregamento e preprocessamento dos dados
    logger.info("2. Carregando e processando dataset...")
    df = pd.read_excel(dataset_path)
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Validação e limpeza dos dados
    df = validate_dataframe(df, ['texto', 'carta', 'evento', 'secao', 'tema'])
    logger.info(f"Dados após validação/limpeza: {len(df)} registros válidos")

    # Estatísticas dos dados
    logger.info(f"Cartas únicas: {df['carta'].nunique()}")
    logger.info(f"Comprimento médio: {df['texto'].str.len().mean():.1f} caracteres")

    # Etapa 3: Divisão estratificada dos dados
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['carta'])
    logger.info(f"Dataset dividido: {len(train_df)} treino, {len(val_df)} validação")

    # Etapa 4: Preparação dos dados para fine-tuning
    logger.info("3. Preparando dados estruturados para fine-tuning...")
    train_data = prepare_data_for_fine_tuning(train_df, PESSOA_X_FINE_TUNING_PROMPT)
    val_data = prepare_data_for_fine_tuning(val_df, PESSOA_X_FINE_TUNING_PROMPT)

    # Criação dos datasets do Hugging Face
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    logger.info(f"Datasets criados: {len(train_dataset)} treino, {len(val_dataset)} validação")

    # Etapa 5: Configuração do modelo e tokenizador
    logger.info("4. Carregando modelo base e configurando quantização...")
    model_name = model_config['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuração da quantização 4-bit
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
    logger.info(f"Modelo carregado: {model_name} com quantização 4-bit")
    
    # Preparação do modelo para treinamento com quantização
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Etapa 6: Configuração dos adaptadores LoRA
    logger.info("5. Configurando LoRA...")
    lora_config = LoraConfig(**lora_config_data)
    model = get_peft_model(model, lora_config)
    logger.info("Parâmetros treináveis configurados")
    model.print_trainable_parameters()

    # Etapa 7: Tokenização dos dados
    logger.info("6. Tokenizando dados...")
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
    logger.info(f"Tokenização concluída: {len(tokenized_train_dataset)} treino, {len(tokenized_val_dataset)} validação")

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

    # Etapa 9: Configuração dos argumentos de treinamento
    logger.info("8. Configurando argumentos de treinamento...")
    training_args = TrainingArguments(**training_args_data)
    logger.info(f"Batch efetivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    # Etapa 10: Inicialização do Trainer
    logger.info("9. Inicializando Trainer...")
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_data['patience'], 
        early_stopping_threshold=early_stopping_data['threshold']
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=curried_compute_metrics,
        callbacks=[early_stopping],
    )
    logger.info("Trainer configurado com sucesso")

    # Etapa 11: Execução do treinamento
    logger.info("10. INICIANDO TREINAMENTO...")
    if torch.cuda.is_available():
        logger.info(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.cuda.empty_cache()

    # Execução do treinamento
    training_result = trainer.train()
    logger.info("=== TREINAMENTO CONCLUÍDO ===")

    # Etapa 12: Avaliação final do modelo
    logger.info("11. Executando avaliação final...")
    logger.info("⏳ Calculando métricas ROUGE e BLEU... (pode demorar 2-3 minutos)")
    final_metrics = trainer.evaluate()
    logger.info("✅ Avaliação final concluída!")

    # Etapa 13: Compilação do relatório de treinamento
    logger.info("12. Compilando relatório de treinamento...")
    training_metrics = {
        'train_runtime': training_result.metrics.get('train_runtime'),
        'train_samples_per_second': training_result.metrics.get('train_samples_per_second'),
        'total_flos': training_result.metrics.get('total_flos', 0),
        'train_loss': training_result.metrics.get('train_loss')
    }

    # Estrutura do relatório final
    final_report = {
        'timestamp': timestamp,
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
            'avg_text_length': float(df['texto'].str.len().mean())
        },
        'model_config': {
            'model_name': model_name,
            'lora_r': lora_config.r,
            'lora_alpha': lora_config.lora_alpha,
            'max_length': model_config['max_length'],
            'epochs_completed': training_args.num_train_epochs
        },
        'training_metrics': training_metrics,
        'final_eval_metrics': final_metrics
    }

    # Salvamento do relatório
    logger.info("⏳ Salvando relatório em JSON...")
    metrics_path = f"./results/training_metrics_{timestamp}.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.info("✅ Relatório salvo!")
    
    # Exibição dos resultados finais
    logger.info("=== RESULTADOS FINAIS ===")
    logger.info(f"📊 ROUGE-L Final: {final_metrics.get('eval_rougeL', 0.0):.4f}")
    logger.info(f"📊 BLEU Final: {final_metrics.get('eval_bleu', 0.0):.4f}")
    logger.info(f"📊 Loss Final: {final_metrics.get('eval_loss', 0.0):.4f}")
    logger.info(f"📁 Métricas detalhadas salvas em: {metrics_path}")

    # Etapa 14: Salvamento do modelo treinado
    logger.info("13. Salvando modelo treinado...")
    logger.info("⏳ Salvando adaptadores LoRA... (pode demorar 1-2 minutos)")
    save_path = "./model_save/lora_model_462_optimized"
    trainer.model.save_pretrained(save_path)

    logger.info("⏳ Salvando tokenizer...")
    tokenizer.save_pretrained(save_path)

    logger.info(f"✅ Modelo salvo com sucesso em: {save_path}")
    logger.info("🎉 === FINE-TUNING FINALIZADO COM SUCESSO === 🎉")

except KeyboardInterrupt:
    logger.info("⚠️ Treinamento interrompido pelo usuário (Ctrl+C)")
    logger.info("💾 Salvando checkpoint de emergência...")
    try:
        emergency_path = f"./model_save/emergency_checkpoint_{timestamp}"
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
    logger.error("💡 Reduza o batch_size ou gradient_accumulation_steps no config YAML")
    
except Exception as e:
    logger.error(f"❌ ERRO CRÍTICO: {str(e)}", exc_info=True)
    logger.error("🔍 Verifique os logs acima para mais detalhes do erro")
    
finally:
    # Limpeza final
    logger.info(f"📝 Log completo salvo em: {log_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cache da GPU limpo")