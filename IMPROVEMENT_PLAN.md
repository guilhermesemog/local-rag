# Plano de evolução do LocalRAG

## Objetivo

Evoluir o LocalRAG de um protótipo funcional para uma aplicação confiável, testada e apresentável como projeto de portfólio.

As mudanças serão realizadas em etapas pequenas. Cada etapa deve ser implementada, revisada e testada manualmente pelo usuário antes do início da etapa seguinte. Nenhuma etapa posterior depende da continuidade automática sem aprovação após a validação manual.

## Estado atual resumido

O projeto já possui:

- API FastAPI;
- interface web HTML, CSS e JavaScript;
- upload de PDF por caminho local e por arquivo enviado;
- parsing e divisão de texto;
- embeddings com Sentence Transformers;
- armazenamento em SQLite ou PostgreSQL com pgvector;
- provedores Gemini, Ollama e modelos Transformers;
- documentação básica no README.

As principais limitações atuais são:

- ausência de testes automatizados e CI;
- validação e tratamento de erros incompletos;
- referências temporárias inseuras no upload;
- erros no JavaScript da interface;
- ausência de metadados e citações;
- pipeline limitado a texto extraído de PDF;
- banco de dados, ciclo de vida e dependências sem organização adequada para produção;
- interface sem recursos modernos de interação com RAG;
- ausência de avaliação quantitativa da qualidade das respostas.

## Regras gerais de implementação

1. Uma etapa somente será considerada concluída após passar pelos testes definidos nela.
2. Mudanças dentro de uma etapa devem permanecer coesas e pequenas.
3. Bugs encontrados durante uma etapa serão corrigidos antes da próxima.
4. Não serão feitas migrações de arquitetura fora da etapa planejada.
5. Alterações em arquivos compartilhados devem preservar compatibilidade com SQLite e PostgreSQL.
6. Credenciais, arquivos locais e dados pessoais não devem ser incluídos no repositório.
7. A interface não deve depender de erros JavaScript silenciosos ou variáveis globais implícitas.
8. Cada etapa deve atualizar a documentação diretamente relacionada.

---

# Etapa 1 — Corrigir a interface web atual

**Prioridade:** P0  
**Dependências:** nenhuma

## Objetivo

Fazer a interface atual carregar e executar suas ações principais sem erros JavaScript.

## Melhorias

1. Remover ou corrigir referências a elementos inexistentes:
   - `geminiAskButton`;
   - `llamaAskButton`.
2. Declarar todas as variáveis JavaScript explicitamente.
3. Substituir `innerHTML` por criação segura de elementos ou sanitização antes da exibição.
4. Melhorar estados de carregamento, sucesso e erro.
5. Impedir envio de perguntas vazias.
6. Impedir upload sem arquivo selecionado.
7. Preparar a interface para que o botão de consulta seja habilitado ou desabilitado corretamente.

## Arquivos possivelmente afetados

- `web/script.js`
- `web/index.html`
- `web/index.css`

## Entrega esperada

- Interface inicial sem erros no console.
- Upload por arquivo funcionando.
- Upload por caminho funcionando.
- Consulta funcionando.
- Limpeza do banco funcionando.
- Resposta exibida como texto seguro.
- Mensagens de erro legíveis.

## Teste manual obrigatório

1. Abrir a aplicação sem documentos enviados.
2. Confirmar que não aparece erro no console.
3. Confirmar que o estado vazio está correto.
4. Selecionar um PDF e realizar upload.
5. Fazer uma pergunta válida.
6. Comparar a resposta com o conteúdo do PDF.
7. Limpar o banco.
8. Confirmar que o estado vazio é restaurado.

## Critério de conclusão

A etapa será concluída somente quando todos os fluxos básicos forem executados manualmente sem erro no console ou comportamento inconsistente.

---

# Etapa 2 — Validar uploads e arquivos temporários

**Prioridade:** P0  
**Dependências:** Etapa 1 concluída

## Objetivo

Evitar colisões de arquivos temporários e impedir uploads inválidos ou excessivamente grandes.

## Melhorias

1. Remover o padrão `/tmp/{nome-do-arquivo}`.
2. Criar arquivos temporários com nome exclusivo.
3. Garantir remoção em caso de sucesso, exceção ou interrupção.
4. Validar a extensão `.pdf`.
5. Validar o tipo MIME informado.
6. Validar assinatura de PDF antes do processamento.
7. Definir limite máximo de tamanho.
8. Retornar mensagens HTTP apropriadas para arquivo ausente, inválido ou muito grande.
9. Avaliar a remoção ou restrição do endpoint que recebe caminho local.

## Arquivos possivelmente afetados

- `src/app.py`
- `src/services.py`
- `src/document_parser.py`
- arquivo de configuração de aplicação, caso seja criado

## Entrega esperada

- Upload simultâneo com nomes iguais não sobrescreve arquivos.
- Arquivos temporários não permanecem após falhas.
- PDFs inválidos são rejeitados antes da ingestão.
- Arquivos maiores que o limite são rejeitados.
- O usuário recebe erros claros.

## Teste manual obrigatório

1. Enviar dois PDFs com o mesmo nome simultaneamente.
2. Enviar um arquivo `.txt` renomeado para `.pdf`.
3. Enviar um PDF vazio ou corrompido.
4. Enviar um PDF acima do limite definido.
5. Repetir o upload de um PDF válido.
6. Confirmar que não restam arquivos temporários.

## Critério de conclusão

Os uploads válidos continuam funcionando e todos os casos inválidos são rejeitados de forma previsível.

---

# Etapa 3 — Padronizar erros e estados da API

**Prioridade:** P0  
**Dependências:** Etapa 2 concluída

## Objetivo

Eliminar falhas genéricas e tornar a API previsível para a interface e para futuras integrações.

## Melhorias

1. Criar exceções internas específicas para:
   - documento vazio;
   - PDF inválido;
   - falha de embedding;
   - banco indisponível;
   - falha no provedor de LLM;
   - recurso não encontrado.
2. Mapear exceções internas para códigos HTTP apropriados.
3. Remover respostas de erro baseadas em `print` ou exceções genéricas.
4. Evitar expor detalhes internos ou credenciais nas respostas.
5. Padronizar o formato de respostas de erro.
6. Adicionar logging estruturado básico.
7. Revisar os retornos dos serviços para não misturar dados, mensagens vazias e erros.

## Arquivos possivelmente afetados

- `src/app.py`
- `src/services.py`
- `src/embedding_engine.py`
- `src/db/postgres_db.py`
- `src/document_parser.py`
- `src/llm/`

## Entrega esperada

- Erros esperados possuem status HTTP coerente.
- Logs mostram a causa sem vazamento de dados sensíveis.
- Respostas da API possuem formato previsível.
- Falhas em serviços podem ser tratadas pela interface.

## Teste manual obrigatório

1. Consultar sem documentos no banco.
2. Desativar ou configurar incorretamente o PostgreSQL.
3. Configurar uma chave Gemini inválida.
4. Parar o Ollama.
5. Enviar um PDF vazio.
6. Confirmar os status, mensagens e logs de cada cenário.

## Critério de conclusão

A API rejeita entradas inválidas e relata falhas externas de modo consistente, sem exceções genéricas ou dados sensíveis. Etapa validada manualmente pelo usuário em 2026-11-19.

---

# Etapa 4 — Introduzir testes automatizados

**Prioridade:** P0  
**Dependências:** Etapas 1 a 3 concluídas

## Objetivo

Criar uma rede de segurança antes das mudanças estruturais no RAG.

## Melhorias

1. Substituir o script manual por testes organizados com pytest.
2. Criar testes para:
   - parsing e limpeza;
   - divisão em chunks;
   - geração de embeddings;
   - inserção e busca no SQLite;
   - inserção e busca no PostgreSQL, quando disponível;
   - upload válido e inválido;
   - consulta vazia;
   - consulta com documentos;
   - limpeza do banco;
   - classes de LLM com mocks.
3. Definir fixtures para PDFs, embeddings e banco temporário.
4. Isolar chamadas externas e modelos pesados durante os testes.
5. Substituir o teste de script por teste unitário ou integração controlada.

## Estrutura sugerida

```text
tests/
  conftest.py
  test_api.py
  test_document_parser.py
  test_embedding_engine.py
  test_services.py
  test_sqlite_db.py
  test_postgres_db.py
```

## Entrega esperada

- Comandos simples para executar testes.
- Cobertura dos principais fluxos.
- Testes rápidos que não dependam necessariamente de rede ou GPU.
- Testes de integração separados dos testes unitários.

## Teste manual obrigatório

1. Executar a suíte completa localmente.
2. Confirmar que todos os testes passam.
3. Fazer uma alteração deliberadamente incorreta em uma função pequena e confirmar que o teste correspondente falha.
4. Restaurar a alteração e repetir a suíte.

## Critério de conclusão

Todos os principais comportamentos possuem testes reproduzíveis e a suíte pode ser executada com um único comando. Validada pelo usuário em 2026-11-19: 10 testes passaram, 1 teste PostgreSQL opcional foi ignorado e 1 aviso de falha foi corrigido antes da confirmação final.

---

# Etapa 5 — Configurar CI inicial

**Prioridade:** P0  
**Dependências:** Etapa 4 concluída

## Objetivo

Garantir que alterações futuras não quebrem os principais comportamentos automaticamente.

## Melhorias

1. Criar workflow de GitHub Actions.
2. Configurar pelo menos duas verificações:
   - lint/formatação;
   - testes automatizados.
3. Utilizar cache de dependências.
4. Configurar Python compatível com o projeto.
5. Evitar downloads de modelos grandes no CI por padrão.
6. Executar a configuração com pull requests e pushes na branch principal.
7. Bloquear merge por falha nas verificações, se aplicável ao repositório.

## Entrega esperada

- Workflow visível no GitHub Actions.
- CI informa lint e testes.
- Falha em um teste gera resultado claro.

## Teste manual obrigatório

1. Executar o workflow pela primeira vez.
2. Abrir um pull request ou alteração pequena em branch.
3. Confirmar que lint e testes são executados.
4. Introduzir deliberadamente uma falha pequena em uma branch de teste.
5. Confirmar que o CI falha.

## Critério de conclusão

O CI protege o projeto contra regressões básicas antes da próxima mudança estrutural. Workflow implementado; validar sua primeira execução no GitHub antes de considerar a etapa concluída.

---

# Etapa 6 — Metadados e citações

**Prioridade:** P1  
**Dependências:** Etapas 1 a 5 concluídas

## Objetivo

Permitir que o usuário saiba quais documentos e páginas foram utilizados em uma resposta.

## Melhorias

1. Adicionar metadados no momento da extração:
   - nome do arquivo;
   - identificador do documento;
   - número da página;
   - número do chunk;
   - título ou nome original.
2. Persistir os metadados no SQLite e PostgreSQL.
3. Trocar retornos de tupla implícitos por objetos ou dataclasses.
4. Retornar fontes junto com a resposta da API.
5. Exibir páginas ou trechos utilizados na interface.
6. Permitir associação de vários documentos por documento enviado.

## Arquivos possivelmente afetados

- `src/services.py`
- `src/document_parser.py`
- `src/db/sqlite_db.py`
- `src/db/postgres_db.py`
- `src/db/vector_db.py`
- `src/app.py`
- `web/`

## Entrega esperada

- Resposta indica as fontes recuperadas.
- Cada fonte contém documento e página.
- A interface mostra texto ou metadados da fonte.
- SQLite e PostgreSQL mantêm o mesmo contrato lógico.

## Teste manual obrigatório

1. Fazer upload de um PDF conhecido com pelo menos três páginas.
2. Fazer uma pergunta relacionada à página 2.
3. Confirmar que a fonte exibida corresponde à página correta.
4. Fazer pergunta relacionada a outro documento.
5. Confirmar que os metadados não misturam documentos.

## Critério de conclusão

Uma resposta pode ser rastreada até o conteúdo e página utilizados. Implementação e testes automatizados e manuais validados pelo usuário em 2026-11-20. Registros anteriores sem metadados permanecem consultáveis, mas não têm fonte identificável.

---

# Etapa 7 — Melhorar parsing e chunking

**Prioridade:** P1  
**Dependências:** Etapa 6 concluída

## Objetivo

Melhorar a qualidade do conteúdo recuperado sem mudar o modelo de vetorização.

## Melhorias

1. Melhorar a limpeza sem destruir espaços ou tabelas.
2. Configurar tamanho e overlap de chunks por configuração.
3. Testar diferentes tamanhos de chunk.
4. Remover chunks vazios e duplicados.
5. Separar texto por página antes do chunking.
6. Adicionar suporte opcional a OCR.
7. Tratar PDFs escaneados de forma explícita.
8. Registrar o método usado para extração em metadados.
9. Validar documentos com pouca ou nenhuma informação textual.

## Entrega esperada

- Chunking configurável.
- Tabelas e quebras comuns são preservadas ou tratadas.
- PDFs escaneados podem ser processados quando OCR estiver disponível.
- O conteúdo enviado ao embedding é previsível e rastreável.

## Teste manual obrigatório

1. Usar um PDF com texto normal.
2. Usar um PDF com tabelas.
3. Usar um PDF com várias colunas.
4. Usar um PDF escaneado, se houver OCR.
5. Comparar respostas antes e depois da mudança usando os mesmos documentos.

## Critério de conclusão

O parsing produz chunks legíveis, sem conteúdo vazio ou excessivamente fragmentado. Testes automatizados e validação manual de PDFs com texto e OCR confirmados pelo usuário em 2026-11-20.

---

# Etapa 8 — Avaliação do RAG

**Prioridade:** P1  
**Dependências:** Etapas 6 e 7 concluídas

## Objetivo

Permitir demonstrar objetivamente a qualidade da solução e identificar regressões.

## Melhorias

1. Criar conjunto pequeno de documentos de avaliação.
2. Criar perguntas com resposta esperada e referência de página.
3. Medir recall dos documentos/chunks recuperados.
4. Medir fidelidade da resposta ao contexto.
5. Avaliar respostas negativas quando a informação não está presente.
6. Registrar tempo de embedding e geração.
7. Comparar diferentes tamanhos de chunk e modelos de embedding.
8. Adicionar relatório resumido ao README.

## Entrega esperada

- Um comando de avaliação reproducível.
- Resultados em formato legível.
- Dados de recall, fidelidade e latência.
- Casos conhecidos de falha documentados.

## Teste manual obrigatório

1. Executar o conjunto de avaliação.
2. Revisar manualmente algumas respostas certas e erradas.
3. Confirmar que a métrica utilizada não depende apenas de texto igual.
4. Comparar resultado com uma configuração anterior.

## Critério de conclusão

É possível afirmar a qualidade do RAG com um conjunto verificável em vez de apenas uma demonstração informal. Benchmark executado, respostas revisadas e configurações comparadas; validação manual e automatizada confirmada pelo usuário em 2026-11-20. A cobertura local por termos é indicativa e não substitui avaliação semântica.

---

# Etapa 9 — Melhorar o gerenciamento do banco

**Prioridade:** P1  
**Dependências:** Etapas 1 a 8 concluídas

## Objetivo

Tornar o acesso aos bancos mais previsível e adequado a uso prolongado.

## Melhorias

1. Criar uma interface única e estável para SQLite e PostgreSQL.
2. Padronizar modelos de metadados e resultados.
3. Gerenciar fechamento de conexões.
4. Utilizar pool de conexões no PostgreSQL.
5. Tornar operações novas com comportamento transacional.
6. Revisar o uso de vetores e estado em memória do SQLite.
7. Preparar migrações de esquema.
8. Adicionar índice apropriado e estratégia de dimensão configurável.
9. Evitar que troca de modelo de embedding cause incompatibilidade silenciosa.

## Entrega esperada

- Ambos os bancos implementam o mesmo contrato.
- Não há estado oculto divergente.
- A troca de banco não exige mudança no fluxo principal.
- Schemas podem evoluir com migrações.

## Teste manual obrigatório

1. Utilizar SQLite e executar upload, consulta e limpeza.
2. Utilizar PostgreSQL e executar os mesmos fluxos.
3. Reiniciar a aplicação e confirmar persistência.
4. Trocar o banco pela configuração e reiniciar.
5. Confirmar que não há consultas incompatíveis.

## Critério de conclusão

SQLite e PostgreSQL oferecem o mesmo comportamento funcional e persistência adequada.

---

# Etapa 10 — Simplificar dependências e configuração

**Prioridade:** P2  
**Dependências:** Etapa 9 concluída

## Objetivo

Reduzir atrito de instalação e tornar a configuração explícita.

## Melhorias

1. Substituir congelamento completo por arquivos separados:
   - `requirements.txt` para execução básica;
   - `requirements-dev.txt`;
   - dependências opcionais para Ollama, Gemini e Transformers.
2. Definir versões mínimas ou intervalos compatíveis.
3. Validar variáveis de ambiente no início.
4. Remover configuração ambígua de padrões.
5. Centralizar modelos e URLs em configurações.
6. Criar `.env.example` completo.
7. Documentar GPU, CPU e provedores disponíveis.

## Entrega esperada

- Instalação básica não força pacotes CUDA desnecessários.
- Configuração inválida falha de forma clara.
- Ambiente SQLite pode ser executado sem PostgreSQL.
- Ambiente de produção não depende de valores secretos no código.

## Teste manual obrigatório

1. Instalar em ambiente virtual novo.
2. Executar somente com SQLite.
3. Executar com PostgreSQL.
4. Executar com Gemini.
5. Executar com Ollama.
6. Executar somente com modelo local Transformers.
7. Remover uma variável obrigatória e confirmar mensagem clara.

## Critério de conclusão

As opções de execução estão documentadas e podem ser iniciadas em ambiente limpo.

---

# Etapa 11 — Docker Compose completo

**Prioridade:** P2  
**Dependências:** Etapas 9 e 10 concluídas

## Objetivo

Permitir iniciar uma demonstração completa com poucos comandos.

## Melhorias

1. Criar Dockerfile da API.
2. Criar serviço FastAPI no Compose.
3. Manter PostgreSQL/pgvector no Compose.
4. Servir a interface web em container ou integração no FastAPI.
5. Tornar Ollama opcional.
6. Adicionar healthchecks.
7. Remover credenciais padrão.
8. Criar volumes apropriados.
9. Documentar comandos de inicialização e encerramento.

## Entrega esperada

- Um comando inicia a aplicação e o banco.
- Endpoints estão disponíveis em porta documentada.
- Interface web está disponível no mesmo ambiente.
- `docker compose down` não remove dados por engano.
- O ambiente pode ser reiniciado sem configuração manual extensa.

## Teste manual obrigatório

1. Iniciar o ambiente em máquina limpa.
2. Acessar interface e documentação da API.
3. Fazer upload e consulta.
4. Reiniciar containers.
5. Confirmar persistência do banco.
6. Encerrar a aplicação.

## Critério de conclusão

A demonstração pode ser iniciada e reiniciada por Docker com poucos comandos.

---

# Etapa 12 — Processamento assíncrono

**Prioridade:** P2  
**Dependências:** Etapas 2, 3 e 11 concluídas

## Objetivo

Evitar bloquear requisições durante uploads grandes e geração longa de respostas.

## Melhorias

1. Migrar operações de I/O e CPU para execução assíncrona ou workers.
2. Implementar status de processamento de documentos.
3. Permitir consulta do progresso.
4. Evitar timeout de upload.
5. Preparar integração com fila caso necessário.
6. Adicionar cancelamento quando possível.

## Entrega esperada

- Upload não bloqueia indefinidamente o servidor.
- Usuário recebe identificador de processamento.
- Estado final do documento pode ser consultado.

## Teste manual obrigatório

1. Enviar PDF grande.
2. Confirmar resposta imediata de processamento iniciado.
3. Consultar estado até conclusão.
4. Fazer uma consulta com outro documento durante o processamento.
5. Confirmar que falhas são visíveis.

## Critério de conclusão

Uploads grandes não travam toda a API e seu progresso é compreensível.

---

# Etapa 13 — Segurança mínima

**Prioridade:** P2  
**Dependências:** Etapas 2, 3 e 11 concluídas

## Objetivo

Reduzir riscos antes de publicar uma demonstração pública.

## Melhorias

1. Remover ou restringir o endpoint de caminho local.
2. Adicionar autenticação para ações administrativas.
3. Proteger a limpeza do banco.
4. Aplicar rate limiting.
5. Revisar CORS.
6. Definir limites de tamanho e duração.
7. Sanitizar todas as entradas exibidas na interface.
8. Não registrar conteúdo completo de documentos ou chaves em logs.
9. Usar secrets gerenciados no ambiente de implantação.

## Entrega esperada

- Operações destrutivas exigem controle adequado.
- Origem e métodos CORS são explícitos.
- Requisições excessivas são limitadas.
- Nenhuma credencial padrão fica ativa.

## Teste manual obrigatório

1. Testar origem não autorizada pela interface.
2. Tentar limpar o banco sem autorização, quando implementada.
3. Testar excesso de consultas.
4. Revisar arquivos enviados e logs.
5. Confirmar que nenhuma chave aparece no repositório ou logs.

## Critério de conclusão

A aplicação possui barreiras básicas contra uso indevido antes de ser publicada.

---

# Etapa 14 — Streaming e melhoria da interface

**Prioridade:** P2  
**Dependências:** Etapas 6, 12 e 13 concluídas

## Objetivo

Transformar a interface em uma experiência moderna de RAG.

## Melhorias

1. Exibir respostas com streaming.
2. Mostrar estado de recuperação de contexto.
3. Adicionar indicador de progresso.
4. Exibir fontes por página.
5. Permitir copiar resposta.
6. Melhorar responsividade.
7. Adicionar estados acessíveis por teclado.
8. Melhorar feedback visual de erro.

## Entrega esperada

- Interface responsiva em desktop e celular.
- Resposta aparece progressivamente.
- Fontes continuam legíveis e relacionadas à conversa.

## Teste manual obrigatório

1. Fazer uma pergunta curta.
2. Fazer uma pergunta longa.
3. Testar celular e desktop.
4. Testar teclado e leitor de tela básico.
5. Confirmar visualização das fontes durante o streaming.

## Critério de conclusão

A interface demonstra claramente o valor do projeto e continua utilizável em diferentes tamanhos.

---

# Etapa 15 — História de conversas e seleção de documentos

**Prioridade:** P2  
**Dependências:** Etapas 6, 12 e 14 concluídas

## Objetivo

Permitir consultas contextuais e comparações entre documentos.

## Melhorias

1. Criar identificador de sessão.
2. Persistir histórico local por sessão.
3. Permitir limpar a conversa separadamente do banco.
4. Permitir selecionar documentos usados em uma consulta.
5. Permitir remover ou substituir documentos.
6. Exibir número de documentos e chunks carregados.
7. Adicionar filtros por documento.

## Entrega esperada

- Conversas persistem entre consultas.
- Usuário escolhe onde procurar.
- Histórico não é confundido com o banco global.

## Teste manual obrigatório

1. Fazer perguntas sucessivas sobre o mesmo assunto.
2. Trocar de documento.
3. Filtrar consulta por documento.
4. Limpar conversa.
5. Limpar banco.
6. Reiniciar aplicação e validar persistência esperada.

## Critério de conclusão

O sistema oferece navegação clara entre documentos e sessões.

---

# Etapa 16 — Publicação da demonstração

**Prioridade:** P2  
**Dependências:** Etapas 11, 13 e 14 concluídas

## Objetivo

Disponibilizar uma demonstração estável para recrutadores e visitantes.

## Melhorias

1. Escolher ambiente de demonstração.
2. Configurar domínio ou URL pública.
3. Criar modo público limitado.
4. Usar modelo local ou provedor gratuito adequado.
5. Definir limites de uso.
6. Adicionar monitoramento básico.
7. Criar screenshots ou GIF curto.
8. Atualizar README com execução, arquitetura e resultados.

## Entrega esperada

- Link funcional no README.
- Demonstração aceita documentos dentro de limite definido.
- Custos externos são controlados.
- Estado público não permite limpeza indiscriminada.

## Teste manual obrigatório

1. Acessar a demonstração fora da máquina local.
2. Fazer upload.
3. Fazer consulta.
4. Enviar múltiplas requisições até o limite.
5. Confirmar mensagem de limite.
6. Confirmar que o ambiente não expõe credenciais.

## Critério de conclusão

A demonstração pode ser usada por outra pessoa sem configuração local.

---

# Ordem final resumida

| Ordem | Etapa | Prioridade | Objetivo |
|---|---|---|---|
| 1 | Correção da interface web | P0 | Remover erros JavaScript e corrigir respostas |
| 2 | Validação de uploads | P0 | Tornar upload seguro e previsível |
| 3 | Padronização de erros | P0 | Estabelecer respostas HTTP consistentes |
| 4 | Testes automatizados | P0 | Cobrir comportamentos essenciais |
| 5 | CI inicial | P0 | Detectar regressões automaticamente |
| 6 | Metadados e citações | P1 | Rastrear respostas |
| 7 | Parsing e chunking | P1 | Melhorar recuperação |
| 8 | Avaliação do RAG | P1 | Medir qualidade |
| 9 | Gerenciamento do banco | P1 | Padronizar SQLite e PostgreSQL |
| 10 | Dependências e configuração | P2 | Facilitar instalação |
| 11 | Docker completo | P2 | Criar demonstração reproduzível |
| 12 | Processamento assíncrono | P2 | Evitar bloqueio em uploads |
| 13 | Segurança mínima | P2 | Preparar implantação pública |
| 14 | Streaming e interface | P2 | Melhorar experiência |
| 15 | Conversas e documentos | P2 | Adicionar gerenciamento de contexto |
| 16 | Publicação | P2 | Disponibilizar portfólio |

# Convenção de andamento

Cada etapa será iniciada somente após o usuário confirmar manualmente que a anterior está concluída.

Ao iniciar uma etapa, será produzido:

- resumo do que será alterado;
- lista de arquivos envolvidos;
- mudanças implementadas;
- testes executados;
- testes manuais que o usuário deve realizar;
- problemas restantes conhecidos;
- solicitação de aprovação antes da próxima etapa.
