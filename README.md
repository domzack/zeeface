# Zeeface

Zeeface é um programa de reconhecimento facial que utiliza as bibliotecas OpenCV, MTCNN e DeepFace. Ele permite detectar rostos em imagens, reconhecer identidades com base em um banco de referências e salvar rostos não identificados para análise posterior.

## Funcionalidades

- **Detecção de Faces**: Detecta rostos em imagens usando a biblioteca MTCNN.
- **Reconhecimento Facial**: Reconhece identidades com base em um banco de referências utilizando modelos como VGG-Face, Facenet, ArcFace e Dlib.
- **Normalização de Nomes**: Renomeia imagens em diretórios para um padrão uniforme.
- **Extração de Faces**: Recorta rostos detectados em imagens e os salva em um diretório específico.
- **Armazenamento de Rostos Não Identificados**: Salva rostos não reconhecidos em uma pasta específica para análise posterior.

## Estrutura de Arquivos

A estrutura de diretórios do projeto é a seguinte:

```
src/
├── referencia/          # Banco de imagens de referência (subpastas por identidade)
├── imagens_processadas/ # Imagens processadas com detecção e reconhecimento
├── recortar/            # Imagens para recortar rostos
├── imagens_teste/       # Imagens de teste para detecção e reconhecimento
├── faces/               # Rostos recortados e não identificados
```

## Funções Principais

### `verificar_banco_referencias(caminho_referencias)`
Verifica se o banco de referências existe e não está vazio.

### `listar_imagens(caminho_imagens, extensoes_validas)`
Lista todas as imagens em um diretório com extensões válidas.

### `criar_diretorio(caminho)`
Cria um diretório caso ele não exista.

### `detectar_faces(imagem)`
Detecta rostos em uma imagem usando MTCNN.

### `reconhecer_face(nome_face, caminho_referencias, modelos)`
Reconhece uma face com base no banco de referências utilizando modelos de reconhecimento facial.

### `normalizar_nomes(caminho_origem, prefixo)`
Renomeia imagens em um diretório para um padrão uniforme e converte para `.jpg` se necessário.

### `normalizar_referencias(caminho_referencia)`
Renomeia imagens em subdiretórios de referência para o padrão `{identidade}-01.jpg`.

### `extrair_faces(caminho_origem, caminho_destino)`
Recorta rostos de imagens em um diretório e os salva no diretório de destino.

### `processar_imagem(imagem, faces, caminho_temp, caminho_faces, modelos, caminho_referencias)`
Processa as faces detectadas em uma imagem, reconhecendo identidades ou salvando rostos não identificados.

### `processar_deteccao_face()`
Executa o fluxo completo de detecção e reconhecimento facial em imagens de teste.

## Como Testar

1. **Preparar o Ambiente**:
   - Certifique-se de que as dependências estão instaladas:
     ```bash
     pip install -r requirements.txt
     ```

2. **Organizar os Arquivos**:
   - Coloque as imagens de referência em subpastas dentro de `src/referencia/`. Cada subpasta deve ter o nome da identidade correspondente.
   - Coloque as imagens para teste em `src/imagens_teste/`.

3. **Executar o Programa**:
   - Para detectar e reconhecer faces:
     ```bash
     python index.py
     ```

4. **Outras Funcionalidades**:
   - Normalizar nomes de imagens:
     ```python
     normalizar_nomes(PATH_IMAGES["teste"])
     ```
   - Extrair rostos de imagens:
     ```python
     extrair_faces(PATH_IMAGES["recortar"], PATH_IMAGES["faces"])
     ```

## Outputs

- **Imagens Processadas**:
  - As imagens com detecção e reconhecimento de rostos são salvas em `src/imagens_processadas/`.

- **Rostos Não Identificados**:
  - Rostos que não foram reconhecidos são salvos em `src/faces/` com o nome `nao-identificado-XX.jpg`.

- **Logs no Console**:
  - O programa exibe mensagens no console indicando o progresso, como imagens processadas, rostos detectados e erros encontrados.

## Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias no projeto.

## Licença

Este projeto está licenciado sob a licença MIT.
