# prompts/pessoa_x_prompt.py
# Módulo de templates de prompt para fine-tuning
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Definição dos templates de prompt utilizados no treinamento do modelo

# Template principal para fine-tuning
# Este template estrutura o prompt de entrada para o modelo, incorporando
# as informações contextuais necessárias para gerar interpretações personalizadas
PESSOA_X_FINE_TUNING_PROMPT = (
    "Gere uma interpretação para a carta '{carta}' "
    "em '{evento}', na seção '{secao}', "
    "sobre o tema '{tema}', com o estilo da pessoa X."
)

# Template de prompt padrão do sistema
# Utilizado para configurações gerais do modelo quando necessário
SYSTEM_PROMPT_DEFAULT = "Você é um assistente de IA útil e profissional."

# Área para templates adicionais
# Aqui podem ser adicionados outros templates conforme necessário:
# - Templates para diferentes estilos de interpretação
# - Templates para diferentes contextos ou ocasiões
# - Templates para prompts de sistema específicos