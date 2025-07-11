# Datasets e Conformidade com LGPD

Este documento explica como trabalhar com datasets no sistema, incluindo aspectos de privacidade e conformidade com a LGPD.

## 📋 Formato dos Dados

### Estrutura Obrigatória

O sistema espera um arquivo Excel (`.xlsx`) com as seguintes colunas:

| Coluna | Tipo | Obrigatório | Descrição |
|--------|------|-------------|-----------|
| `evento` | String | ✅ | Contexto do evento (ex: "Pinkpop", "Woodstock") |
| `carta` | String | ✅ | Nome da carta (ex: "Dama", "Rei") |
| `tema` | String | ✅ | Tema da interpretação (ex: "Catppuccin", "Tokyo Night") |
| `secao` | String | ✅ | Seção específica (ex: "Ficção científica", "Biografia") |
| `texto` | String | ✅ | Texto de interpretação esperado |

### Exemplo de Estrutura

```
evento       | carta | tema         | secao             | texto
-------------|-------|--------------|-------------------|------------------
Pinkpop      | Dama  | Catppuccin   | Ficção científica | Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Woodstock    | Rei   | Tokyo Night  | Biografia         | Ut at risus vehicula, feugiat nibh a, venenatis felis.
```

## Conformidade com LGPD

### Por que o Dataset não está no Repositório

O dataset (`database/db_462.xlsx`) **não está incluído** no repositório público pelos seguintes motivos:

1. **Dados Pessoais**: Pode conter interpretações personalizadas
2. **Propriedade Intelectual**: Conteúdo autoral específico
3. **Conformidade LGPD**: Prevenção de exposição desnecessária
4. **Controle de Acesso**: Distribuição controlada

### Princípios da LGPD Aplicados

#### 1. Minimização de Dados
```python
# Apenas colunas necessárias são processadas
required_columns = ['evento', 'carta', 'tema', 'secao', 'texto']
df = validate_dataframe(df, required_columns)
```

#### 2. Finalidade Específica
- **Finalidade**: Treinamento de modelo de IA
- **Uso**: Geração de interpretações personalizadas
- **Retenção**: Dados mantidos apenas durante o treinamento

#### 3. Transparência
- Processamento documentado
- Código fonte disponível
- Logs detalhados do treinamento

#### 4. Segurança
- Dados processados localmente
- Sem transmissão para serviços externos
- Ambiente controlado

## Preparação de Dados

### Validação Automática

O sistema aplica validações automáticas:

```python
def validate_dataframe(df, required_columns):
    # Verificação de colunas obrigatórias
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Colunas necessárias: {required_columns}")
    
    # Remoção de valores nulos
    df = df.dropna(subset=required_columns)
    
    # Limpeza de texto
    df['texto'] = df['texto'].apply(clean_text)
    
    return df
```

### Limpeza de Dados

```python
def clean_text(text):
    """Remove espaços múltiplos e caracteres desnecessários"""
    if pd.isna(text):
        return None
    return re.sub(r'\s+', ' ', str(text)).strip()
```

## Configuração do Dataset

### Caminho do Arquivo

```yaml
# config/training_config.yaml
dataset_path: "./database/db_462.xlsx"
```

### Divisão dos Dados

```yaml
test_size: 0.15      # 15% para validação
random_state: 42     # Seed para reprodutibilidade
```

**Divisão estratificada:**
- Mantém proporção de cartas em treino/validação
- Garante representatividade
- Evita vazamento de dados

## Qualidade dos Dados

### Métricas de Qualidade

```python
# Estatísticas geradas automaticamente
print(f"Total de registros: {len(df)}")
print(f"Cartas únicas: {df['carta'].nunique()}")
print(f"Eventos únicos: {df['evento'].nunique()}")
print(f"Comprimento médio: {df['texto'].str.len().mean():.1f}")
print(f"Comprimento mínimo: {df['texto'].str.len().min()}")
print(f"Comprimento máximo: {df['texto'].str.len().max()}")
```

### Distribuição Recomendada

| Métrica | Valor Recomendado | Motivo |
|---------|-------------------|--------|
| **Registros por carta** | 15-30 | Diversidade suficiente |
| **Comprimento médio** | 100-300 caracteres | Qualidade vs. eficiência |
| **Variação de eventos** | 5+ eventos | Generalização |
| **Variação de temas** | 10+ temas | Cobertura ampla |

## Segurança dos Dados

### Armazenamento Local

```bash
# Estrutura de pastas
database/
├── db_462.xlsx           # Dataset principal (não versionado)
├── .gitignore           # Ignora arquivos de dados
└── sample_structure.xlsx # Exemplo de estrutura (versionado)
```

### Arquivo .gitignore

```gitignore
# Dados sensíveis
database/*.xlsx
database/*.csv
database/*.json
!database/sample_structure.xlsx

# Resultados de treinamento
results/training_log_*.log
results/training_metrics_*.json

# Modelos treinados
model_save/*/
```

## Boas Práticas

### Anonimização

Se necessário, você pode implementar anonimização dos dados:

- Remoção de nomes próprios
- Substituição de informações pessoais
- Generalização de dados específicos

Isso deve ser feito antes do treinamento, modificando o dataset original conforme suas necessidades de privacidade.

### Backup Seguro

```bash
# Backup com criptografia
gpg --cipher-algo AES256 --compress-algo 2 --symmetric database/db_462.xlsx

# Backup em local seguro
cp database/db_462.xlsx.gpg /backup/secure/
```

### Controle de Versão

```bash
# Versionamento de datasets
database/
├── db_462_v1.xlsx
├── db_462_v2.xlsx
└── db_462_current.xlsx -> db_462_v2.xlsx
```

## 📋 Checklist de Conformidade

### Antes do Treinamento

- [ ] Dataset contém apenas dados necessários
- [ ] Dados foram validados e limpos
- [ ] Finalidade do uso está documentada
- [ ] Acesso é controlado e auditado
- [ ] Backup seguro foi criado

### Durante o Treinamento

- [ ] Processamento é local (sem upload)
- [ ] Logs não expõem dados sensíveis
- [ ] Métricas são agregadas
- [ ] Acesso é monitorado

### Após o Treinamento

- [ ] Dados temporários são removidos
- [ ] Modelo não "memoriza" dados específicos
- [ ] Logs são revisados
- [ ] Documentação é atualizada

## Atualização de Datasets

### Processo Seguro

1. **Validação**: Verificar nova estrutura
2. **Backup**: Preservar versão anterior
3. **Teste**: Executar com amostra pequena
4. **Validação**: Confirmar compatibilidade
5. **Aplicação**: Usar dataset completo

### Script de Migração

Para migrar datasets entre versões:

1. **Backup**: Sempre faça backup do dataset original
2. **Validação**: Verifique se a nova estrutura é compatível
3. **Teste**: Execute com uma amostra pequena primeiro
4. **Aplicação**: Processe o dataset completo
5. **Verificação**: Confirme que o sistema funciona com os novos dados

Use as funções existentes `validate_dataframe` e `clean_text` para garantir consistência.

## Suporte e Dúvidas

### Questões Comuns

**Q: Posso usar dados de terceiros?**
A: Apenas com autorização expressa e conformidade legal.

**Q: Como garantir a qualidade dos dados?**
A: Use as validações automáticas e revise manualmente amostras.

**Q: É necessário consentimento para uso?**
A: Depende da fonte e finalidade. Consulte assessoria jurídica.

### Contato

Para questões sobre conformidade:
- Consulte assessoria jurídica
- Revise política de privacidade
- Documente todas as decisões

---

**Próximo**: [Troubleshooting](troubleshooting.md)