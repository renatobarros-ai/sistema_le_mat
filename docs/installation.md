# Guia de Instalação

Este guia detalha como configurar o ambiente e instalar todas as dependências necessárias para o sistema de fine-tuning PTT5.

## 📋 Pré-requisitos

### Hardware Mínimo
- **GPU**: NVIDIA com 4GB+ VRAM (recomendado 8GB+)
- **RAM**: 8GB mínimo (16GB+ recomendado)
- **Armazenamento**: 10GB livres
- **CPU**: Qualquer processador moderno

### Software Base
- **Python**: 3.8 ou superior
- **CUDA**: 11.8 ou superior
- **Git**: Para clonar o repositório

## Instalação Passo a Passo

### 1. Verificar Ambiente

Primeiro, verifique se o Python e CUDA estão instalados:

```bash
# Verificar Python
python --version
# Deve retornar Python 3.8+

# Verificar CUDA (se disponível)
nvidia-smi
# Deve mostrar informações da GPU
```

### 2. Clonar o Repositório

```bash
# Clone o repositório
git clone https://github.com/renatobarros-ai/sistema_le_mat.git
cd sistema_le_mat
```

### 3. Criar Ambiente Virtual

**Recomendado**: Use um ambiente virtual para isolar as dependências:

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 4. Instalar Dependências

```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Ou instalar manualmente as principais:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate accelerate peft
pip install pandas openpyxl scikit-learn PyYAML
pip install bitsandbytes rouge-score sacrebleu
```

### 5. Verificar Instalação

Execute estes comandos para verificar se tudo está funcionando:

```bash
# Verificar Python e bibliotecas principais
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"

# Verificar CUDA (se disponível)
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"

# Testar imports do projeto
python -c "from utils.data_processing import validate_dataframe; print('Utils OK')"
python -c "from prompts.pessoa_x_prompt import PESSOA_X_FINE_TUNING_PROMPT; print('Prompts OK')"
```

## Configuração do Ambiente

### Variáveis de Ambiente

O sistema configura automaticamente as seguintes variáveis:

```bash
# Configuradas automaticamente no código
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
```

### Estrutura de Pastas

Após a instalação, certifique-se de que as seguintes pastas existam:

```bash
# Criar pastas necessárias
mkdir -p results
mkdir -p model_save
mkdir -p database
```

## Preparação dos Dados

### Dataset

1. **Formato**: O sistema espera um arquivo Excel (`.xlsx`) com as colunas:
   - `evento`: Contexto do evento
   - `carta`: Nome da carta
   - `tema`: Tema da interpretação
   - `secao`: Seção específica
   - `texto`: Texto de saída esperado

2. **Localização**: Coloque o arquivo em `database/db_462.xlsx`

3. **Exemplo de estrutura**:
   ```
   evento       | carta     | tema          | secao               | texto
   Pinkpop      | Dama      | Catppuccin    | Ficção científica   | Lorem ipsum dolor sit amet, consectetur adipiscing elit.
   Woodstock    | Rei       | Tokio Night   | Biografia           | Ut at risus vehicula, feugiat nibh a, venenatis felis.
   ```

### Configuração

O arquivo `config/training_config.yaml` já está configurado com valores otimizados, mas você pode ajustar:

- **Batch size**: Ajuste conforme sua GPU
- **Learning rate**: Modifique se necessário
- **Epochs**: Controle a duração do treinamento

## Problemas Comuns

### Erro de CUDA
```bash
# Se encontrar erro de CUDA, instale a versão correta do PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Erro de Memória
```bash
# Reduza o batch size em training_config.yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
```

### Erro de Dependências
```bash
# Atualize o pip e tente novamente
pip install --upgrade pip
pip install -r requirements.txt
```

## Próximos Passos

Após a instalação bem-sucedida:

1. **Verifique** se o dataset está no formato correto
2. **Configure** os parâmetros em `training_config.yaml`
3. **Execute** o treinamento com `python train_model.py`
4. **Monitore** o progresso nos logs

## Verificação Final

Execute esta checklist antes de iniciar o treinamento:

- [ ] Python 3.8+ instalado
- [ ] CUDA funcionando (se usando GPU)
- [ ] Todas as dependências instaladas
- [ ] Dataset no formato correto
- [ ] Pastas criadas
- [ ] Configuração ajustada

Se todos os itens estiverem marcados, você está pronto para iniciar o treinamento!

## Suporte

Se encontrar problemas durante a instalação:

1. Consulte o [guia de troubleshooting](troubleshooting.md)
2. Verifique se sua versão do Python/CUDA é compatível
3. Abra uma issue no repositório com detalhes do erro

---
**Próximo**: [Configuração do Sistema](configuration.md)
