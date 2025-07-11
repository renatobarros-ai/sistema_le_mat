# train_model.py - VERSÃO FINAL CORRIGIDA
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

# Importações de módulos personalizados
from prompts.tarologa_x_prompt import TAROLOGA_X_FINE_TUNING_PROMPT
from utils.data_processing import prepare_data_for_fine_tuning, tokenize_function, validate_dataframe
from utils.evaluation_metrics import compute_metrics

# =============================================================================
# CONFIGURAÇÃO DE LOGGING E SUPRESSÃO DE WARNINGS
# =============================================================================
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

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
    logger.info("Treinamento interrompido pelo usuário (Ctrl+C)")
    logger.info(f"Log salvo em: {log_path}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# =============================================================================
# FINE-TUNING PRINCIPAL
# =============================================================================
logger.info("=== INICIANDO FINE-TUNING PTT5 - DATASET EXPANDIDO ===")

try:
    # Carregar configurações do arquivo YAML
    logger.info("1. Carregando configurações de treinamento...")
    with open('config/training_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_config = config['model_config']
    lora_config_data = config['lora_config']
    training_args_data = config['training_args']
    early_stopping_data = config['early_stopping']
    dataset_path = config['dataset_path']
    test_size = config['test_size']
    random_state = config['random_state']

    logger.info(f"Configurações carregadas para o modelo: {model_config['name']}")

    # Carregamento e processamento de dados
    logger.info("2. Carregando e processando dataset...")
    df = pd.read_excel(dataset_path)
    logger.info(f"Dados carregados: {len(df)} registros")
    
    # Validação e limpeza de dados
    df = validate_dataframe(df, ['texto', 'arcano_maior', 'evento', 'secao', 'tema'])
    logger.info(f"Dados após validação/limpeza: {len(df)} registros válidos")

    logger.info(f"Arcanos únicos: {df['arcano_maior'].nunique()}")
    logger.info(f"Comprimento médio: {df['texto'].str.len().mean():.1f} caracteres")

    # Divisão estratificada dos dados
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['arcano_maior'])
    logger.info(f"Dataset dividido: {len(train_df)} treino, {len(val_df)} validação")

    # Preparação dos dados para fine-tuning
    logger.info("3. Preparando dados estruturados para fine-tuning...")
    train_data = prepare_data_for_fine_tuning(train_df, TAROLOGA_X_FINE_TUNING_PROMPT)
    val_data = prepare_data_for_fine_tuning(val_df, TAROLOGA_X_FINE_TUNING_PROMPT)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    logger.info(f"Datasets criados: {len(train_dataset)} treino, {len(val_dataset)} validação")

    # Carregamento do modelo base e configuração
    logger.info("4. Carregando modelo base e configurando quantização...")
    model_name = model_config['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configuração de quantização
    bnb_compute_dtype = getattr(torch, model_config['quantization']['bnb_4bit_compute_dtype'].split('.')[-1])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config['quantization']['load_in_4bit'],
        bnb_4bit_use_double_quant=model_config['quantization']['bnb_4bit_use_double_quant'],
        bnb_4bit_quant_type=model_config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=bnb_compute_dtype
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=bnb_compute_dtype
    )
    logger.info(f"Modelo carregado: {model_name} com quantização 4-bit")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # Configuração LoRA
    logger.info("5. Configurando LoRA...")
    lora_config = LoraConfig(**lora_config_data)
    model = get_peft_model(model, lora_config)
    logger.info("Parâmetros treináveis configurados")
    model.print_trainable_parameters()

    # Tokenização dos dados
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

    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=model_config['max_length']
    )

    # Configuração de métricas
    logger.info("7. Configurando métricas de avaliação...")
    def curried_compute_metrics(eval_pred):
        return compute_metrics(eval_pred, tokenizer)

    # Configuração de treinamento
    logger.info("8. Configurando argumentos de treinamento...")
    training_args = TrainingArguments(**training_args_data)
    logger.info(f"Batch efetivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    # Inicialização do Trainer
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

    # Início do treinamento
    logger.info("10. INICIANDO TREINAMENTO...")
    if torch.cuda.is_available():
        logger.info(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        torch.cuda.empty_cache()

    # Treinamento
    training_result = trainer.train()
    logger.info("=== TREINAMENTO CONCLUÍDO ===")

    # Avaliação final
    logger.info("11. Executando avaliação final...")
    logger.info("⏳ Calculando métricas ROUGE e BLEU... (pode demorar 2-3 minutos)")
    final_metrics = trainer.evaluate()
    logger.info("✅ Avaliação final concluída!")

    # Compilação do relatório
    logger.info("12. Compilando relatório de treinamento...")
    training_metrics = {
        'train_runtime': training_result.metrics.get('train_runtime'),
        'train_samples_per_second': training_result.metrics.get('train_samples_per_second'),
        'total_flos': training_result.metrics.get('total_flos', 0),
        'train_loss': training_result.metrics.get('train_loss')
    }

    # Relatório final
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
            'unique_arcanos': df['arcano_maior'].nunique(),
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
    
    # Logging dos resultados finais
    logger.info("=== RESULTADOS FINAIS ===")
    logger.info(f"📊 ROUGE-L Final: {final_metrics.get('eval_rougeL', 0.0):.4f}")
    logger.info(f"📊 BLEU Final: {final_metrics.get('eval_bleu', 0.0):.4f}")
    logger.info(f"📊 Loss Final: {final_metrics.get('eval_loss', 0.0):.4f}")
    logger.info(f"📁 Métricas detalhadas salvas em: {metrics_path}")

    # Salvamento do modelo
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
    logger.info(f"📝 Log completo salvo em: {log_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 Cache da GPU limpo")
