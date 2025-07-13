# prompts/pessoa_x_prompt.py
# Módulo de templates de prompt para fine-tuning - VERSÃO OTIMIZADA
# Autor: Renato Barros
# Email: falecomrenatobarros@gmail.com
# Data: 2025
# Descrição: Templates otimizados para melhor aprendizado do modelo

# Template principal OTIMIZADO para fine-tuning
# Formato direto e natural, sem instruções explícitas
PESSOA_X_FINE_TUNING_PROMPT = (
    "Carta: {carta} | Evento: {evento} | Seção: {secao} | Tema: {tema}"
)

# Template alternativo mais conciso (backup)
PESSOA_X_FINE_TUNING_PROMPT_CONCISO = (
    "{carta} - {evento} - {secao} - {tema}"
)

# Template de prompt padrão do sistema
SYSTEM_PROMPT_DEFAULT = "Você é um assistente de IA útil e profissional."

# Templates para diferentes variações (uso futuro)
PESSOA_X_VARIATIONS = {
    "formato1": "Carta: {carta} | Evento: {evento} | Seção: {secao} | Tema: {tema}",
    "formato2": "{carta} no contexto de {evento}, seção {secao}, tema {tema}",
    "formato3": "[{carta}] {evento} > {secao} > {tema}"
}
