# Sistema de Fine-tuning PTT5 com QLoRA

Um sistema completo para fine-tuning do modelo PTT5 (Portuguese T5) usando quantização 4-bit e adaptadores LoRA, otimizado para geração de texto personalizada.

## Visão Geral

Este projeto implementa um pipeline completo de fine-tuning para o modelo PTT5 base, utilizando técnicas avançadas de otimização como:

- **Quantização 4-bit**: Reduz o uso de memória GPU em ~75%
- **QLoRA**: Combinação de quantização com adaptadores LoRA
- **PEFT (Parameter-Efficient Fine-Tuning)**: Treina apenas uma pequena fração dos parâmetros
- **Métricas robustas**: Avaliação com ROUGE e BLEU

## Início Rápido

### Pré-requisitos
- Python 3.8+
- CUDA 11.8+ (para GPU)
- 8GB+ de RAM
- 4GB+ de VRAM (recomendado)

### Instalação
```bash
# Clone o repositório
git clone https://github.com/renatobarros-ai/sistema_le_mat.git
cd sistema_le_mat

# Instale as dependências
pip install -r requirements.txt
```

### Uso Básico
```bash
# Execute o treinamento
python train_model.py
```

## Estrutura do Projeto

```
sistema_le_mat/
├── config/
│   └── training_config.yaml    # Configurações de treinamento
├── database/
│   └── db_462.xlsx            # Dataset (não versionado)
├── docs/                      # Documentação
├── model_save/                # Modelos treinados
├── prompts/
│   └── pessoa_x_prompt.py     # Templates de prompt
├── results/                   # Logs e métricas
├── utils/
│   ├── data_processing.py     # Processamento de dados
│   └── evaluation_metrics.py  # Métricas de avaliação
├── train_model.py             # Script principal
└── requirements.txt           # Dependências
```

## Documentação Completa

### Configuração e Instalação
- **[Guia de Instalação](docs/installation.md)** - Instalação detalhada e configuração do ambiente
- **[Configuração do Sistema](docs/configuration.md)** - Personalização de parâmetros e configurações

### Arquitetura e Desenvolvimento
- **[Arquitetura do Sistema](docs/architecture.md)** - Visão técnica detalhada da implementação
- **[Processamento de Dados](docs/data-processing.md)** - Como os dados são processados e preparados

### Guias de Uso
- **[Guia de Uso](docs/usage.md)** - Como usar o sistema e interpretar resultados
- **[Troubleshooting](docs/troubleshooting.md)** - Solução de problemas comuns

### Dados e Privacidade
- **[Datasets e LGPD](docs/datasets.md)** - Informações sobre dados e conformidade

## Características Principais

### Otimizado para Eficiência
- **Quantização 4-bit**: Reduz drasticamente o uso de memória
- **LoRA**: Treina apenas 0.1% dos parâmetros do modelo
- **Gradient Checkpointing**: Economia adicional de memória

### Flexível e Configurável
- **Configuração YAML**: Fácil personalização de parâmetros
- **Templates de Prompt**: Sistema modular para diferentes estilos
- **Métricas Robustas**: Avaliação confiável com ROUGE e BLEU

### Pronto para Produção
- **Logging Completo**: Rastreamento detalhado do treinamento
- **Tratamento de Erros**: Sistema robusto de recuperação
- **Salvamento Automático**: Checkpoints e modelos preservados

## Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|------------|---------|-------------|
| **GPU** | 4GB VRAM | 8GB+ VRAM |
| **RAM** | 8GB | 16GB+ |
| **Armazenamento** | 10GB | 20GB+ |
| **CUDA** | 11.8+ | 12.0+ |

## Fluxo de Trabalho

1. **Preparação**: Configurar ambiente e dataset
2. **Configuração**: Ajustar parâmetros no arquivo YAML
3. **Treinamento**: Executar o script principal
4. **Avaliação**: Analisar métricas e resultados
5. **Inferência**: Usar o modelo treinado

## Métricas de Avaliação

O sistema utiliza métricas padrão da área:
- **ROUGE-L**: Medida de qualidade baseada em subsequências
- **BLEU**: Métrica de similaridade com referência
- **Loss**: Perda de treinamento e validação

## Contribuição

Para contribuir com o projeto:
1. Faça um fork do repositório
2. Crie uma branch para sua feature
3. Implemente as mudanças
4. Abra um Pull Request

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE) - veja o arquivo LICENSE para mais detalhes.

## Suporte

- **Issues**: Use o sistema de issues do GitHub
- **Documentação**: Consulte a pasta [docs/](docs/) para informações detalhadas
- **Troubleshooting**: Veja o [guia de solução de problemas](docs/troubleshooting.md)
- **Contato direto**: falecomrenatobarros@gmail.com

---

**Desenvolvido por Renato Barros** - Sistema de Fine-tuning PTT5 com QLoRA

📧 **Contato:** falecomrenatobarros@gmail.com
