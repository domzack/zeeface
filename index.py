# reconhecimento_deepface.py

import cv2
from mtcnn import MTCNN  # Altere para o módulo original
from deepface import DeepFace
import os
import sys

PATH_IMAGES = {
    "referencia": "./src/referencia",  # Banco de imagens de referência
    "processadas": "./src/imagens_processadas",  # Imagens processadas
    "recortar": "./src/recortar",  # Imagens para recortar rostos
    "teste": "./src/imagens_teste",  # Imagens de teste
    "faces": "./src/faces",  # Pasta para salvar rostos recortados
}


def verificar_banco_referencias(caminho_referencias):
    if not os.path.exists(caminho_referencias) or not os.listdir(caminho_referencias):
        print("Banco de referências não encontrado ou vazio.")
        sys.exit(1)


def listar_imagens(caminho_imagens, extensoes_validas):
    return [
        f
        for f in os.listdir(caminho_imagens)
        if os.path.splitext(f)[1].lower() in extensoes_validas
    ]


def criar_diretorio(caminho):
    if not os.path.exists(caminho):
        os.makedirs(caminho)


def detectar_faces(imagem):
    detector = MTCNN()
    return detector.detect_faces(imagem)


def reconhecer_face(nome_face, caminho_referencias, modelos):
    identidade = "Desconhecido"
    for modelo in modelos:
        print(f"Tentando reconhecer com o modelo: {modelo}")
        try:
            resultado = DeepFace.find(
                img_path=nome_face,
                db_path=caminho_referencias,
                enforce_detection=True,
                model_name=modelo,
                detector_backend="mtcnn",
            )

            # Verifica se o resultado é uma lista
            if isinstance(resultado, list) and len(resultado) > 0:
                resultado = resultado[0]  # Pega o primeiro DataFrame

            if not resultado.empty:
                # Extrai o nome do subdiretório (identidade)
                identidade = os.path.basename(
                    os.path.dirname(resultado.iloc[0]["identity"])
                )
                print(f"Face reconhecida como: {identidade} usando o modelo {modelo}")
                break
        except Exception as e:
            print(f"Erro ao usar o modelo {modelo}: {e}")
    return identidade


def limpar_pasta_temp(caminho_temp):
    for temp_file in os.listdir(caminho_temp):
        os.remove(os.path.join(caminho_temp, temp_file))
    os.rmdir(caminho_temp)


def normalizar_nomes(caminho_origem, prefixo="TBB"):
    """
    Renomeia todas as imagens no diretório especificado para um nome padrão e converte para .jpg se necessário.

    Args:
        caminho_origem (str): Caminho do diretório contendo as imagens.
        prefixo (str): Prefixo para os nomes das imagens renomeadas.
    """
    extensoes_validas = [".jpg", ".jpeg", ".png"]
    imagens = listar_imagens(caminho_origem, extensoes_validas)

    if not imagens:
        print("Nenhuma imagem encontrada para renomear.")
        return

    for i, imagem_nome in enumerate(imagens, start=1):
        extensao = os.path.splitext(imagem_nome)[1].lower()
        novo_nome = f"{prefixo}-{i:02d}.jpg"
        caminho_antigo = os.path.join(caminho_origem, imagem_nome)
        caminho_novo = os.path.join(caminho_origem, novo_nome)

        # Converte para .jpg se necessário
        if extensao != ".jpg":
            imagem = cv2.imread(caminho_antigo)
            cv2.imwrite(caminho_novo, imagem)
            os.remove(caminho_antigo)
        else:
            os.rename(caminho_antigo, caminho_novo)

        print(f"Renomeado: {imagem_nome} -> {novo_nome}")


def normalizar_referencias(caminho_referencia):
    """
    Renomeia imagens em subdiretórios de referência para o padrão {identidade}-01.jpg.

    Args:
        caminho_referencia (str): Caminho do diretório contendo as subpastas de identidades.
    """
    if not os.path.exists(caminho_referencia):
        print(f"O caminho {caminho_referencia} não existe.")
        return

    for identidade in os.listdir(caminho_referencia):
        caminho_identidade = os.path.join(caminho_referencia, identidade)
        if not os.path.isdir(caminho_identidade):
            continue

        extensoes_validas = [".jpg", ".jpeg", ".png"]
        imagens = listar_imagens(caminho_identidade, extensoes_validas)

        if not imagens:
            print(f"Nenhuma imagem encontrada para a identidade: {identidade}")
            continue

        for i, imagem_nome in enumerate(imagens, start=1):
            extensao = os.path.splitext(imagem_nome)[1].lower()
            novo_nome = f"{identidade}-{i:02d}.jpg"
            caminho_antigo = os.path.join(caminho_identidade, imagem_nome)
            caminho_novo = os.path.join(caminho_identidade, novo_nome)

            # Converte para .jpg se necessário
            if extensao != ".jpg":
                imagem = cv2.imread(caminho_antigo)
                cv2.imwrite(caminho_novo, imagem)
                os.remove(caminho_antigo)
            else:
                os.rename(caminho_antigo, caminho_novo)

            print(f"Renomeado: {imagem_nome} -> {novo_nome}")


def extrair_faces(caminho_origem, caminho_destino):
    """
    Recorta rostos de imagens no diretório especificado e salva no diretório de destino.

    Args:
        caminho_origem (str): Caminho do diretório contendo as imagens para recorte.
        caminho_destino (str): Caminho do diretório onde os rostos recortados serão salvos.
    """
    criar_diretorio(caminho_destino)
    extensoes_validas = [".jpg", ".jpeg", ".png"]
    imagens = listar_imagens(caminho_origem, extensoes_validas)

    if not imagens:
        print("Nenhuma imagem encontrada para recortar.")
        return

    contador = 1
    for imagem_nome in imagens:
        caminho_imagem = os.path.join(caminho_origem, imagem_nome)
        imagem = cv2.imread(caminho_imagem)

        if imagem is None:
            print(f"Erro ao carregar a imagem: {imagem_nome}")
            continue

        faces = detectar_faces(imagem)
        if not faces:
            print(f"Nenhuma face detectada em {imagem_nome}.")
            continue

        for face in faces:
            x, y, w, h = face["box"]
            face_crop = imagem[y : y + h, x : x + w]

            # Gera um nome único para o arquivo
            while True:
                nome_arquivo = f"face-{contador:02d}.jpg"
                caminho_arquivo = os.path.join(caminho_destino, nome_arquivo)
                if not os.path.exists(caminho_arquivo):
                    break
                contador += 1

            cv2.imwrite(caminho_arquivo, face_crop)
            print(f"Rosto salvo em: {caminho_arquivo}")
            contador += 1

        # Caso nenhuma face seja identificada, salve a imagem original na pasta de faces
        if not faces:
            while True:
                nome_arquivo = f"nao-identificado-{contador:02d}.jpg"
                caminho_arquivo = os.path.join(PATH_IMAGES["faces"], nome_arquivo)
                if not os.path.exists(caminho_arquivo):
                    break
                contador += 1

            cv2.imwrite(caminho_arquivo, imagem)
            print(f"Imagem sem rosto salva em: {caminho_arquivo}")
            contador += 1


def processar_imagem(
    imagem, faces, caminho_temp, caminho_faces, modelos, caminho_referencias
):
    """
    Processa as faces detectadas em uma imagem. Salva as faces não identificadas na pasta de faces.

    Args:
        imagem (numpy.ndarray): Imagem original.
        faces (list): Lista de faces detectadas.
        caminho_temp (str): Caminho para salvar temporariamente as faces.
        caminho_faces (str): Caminho para salvar faces não identificadas.
        modelos (list): Modelos de reconhecimento facial.
        caminho_referencias (str): Caminho do banco de referências.
    """
    criar_diretorio(caminho_faces)

    for i, face in enumerate(faces):
        x, y, w, h = face["box"]
        face_crop = imagem[y : y + h, x : x + w]

        # Salva temporariamente a face para reconhecimento
        nome_face_temp = os.path.join(caminho_temp, f"temp_face_{i+1}.jpg")
        cv2.imwrite(nome_face_temp, face_crop)

        identidade = reconhecer_face(nome_face_temp, caminho_referencias, modelos)

        if identidade != "Desconhecido":
            # Adiciona o nome abaixo do quadro e desenha o quadro azul
            texto_x = x
            texto_y = y + h + 20  # Posiciona o texto abaixo do quadro
            cv2.putText(
                imagem,
                identidade,
                (texto_x, texto_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            # Apenas desenha o quadro vermelho para desconhecidos
            cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Salva a face não identificada na pasta de faces
            contador = 1
            while True:
                nome_arquivo = f"nao-identificado-{contador:02d}.jpg"
                caminho_arquivo = os.path.join(caminho_faces, nome_arquivo)
                if not os.path.exists(caminho_arquivo):
                    break
                contador += 1

            cv2.imwrite(caminho_arquivo, face_crop)
            print(f"Face não identificada salva em: {caminho_arquivo}")


def processar_deteccao_face():
    # Configurações
    CAMINHO_IMAGENS = PATH_IMAGES["teste"]
    CAMINHO_REFERENCIAS = PATH_IMAGES["referencia"]
    CAMINHO_SALVAR_IMAGENS = PATH_IMAGES["processadas"]
    CAMINHO_TEMP = "./temp"
    EXTENSOES_VALIDAS = [".jpg", ".jpeg", ".png"]
    MODELOS = ["VGG-Face", "Facenet", "ArcFace", "Dlib"]

    verificar_banco_referencias(CAMINHO_REFERENCIAS)
    imagens = listar_imagens(CAMINHO_IMAGENS, EXTENSOES_VALIDAS)

    if not imagens:
        print("Nenhuma imagem encontrada no diretório.")
        sys.exit(1)

    criar_diretorio(CAMINHO_SALVAR_IMAGENS)
    criar_diretorio(CAMINHO_TEMP)

    # Adicione uma flag global para controle
    interromper = False

    # Processa cada imagem
    for imagem_nome in imagens:
        if interromper:
            print("Processo interrompido pelo usuário.")
            break

        # print(f"Processando: {imagem_nome}")
        caminho_imagem = os.path.join(CAMINHO_IMAGENS, imagem_nome)
        imagem = cv2.imread(caminho_imagem)

        faces = detectar_faces(imagem)
        if not faces:
            print(f"Nenhuma face detectada em {imagem_nome}.")
            continue

        processar_imagem(
            imagem,
            faces,
            CAMINHO_TEMP,
            PATH_IMAGES["faces"],
            MODELOS,
            CAMINHO_REFERENCIAS,
        )

        # Salva a imagem processada no diretório
        caminho_salvar = os.path.join(
            CAMINHO_SALVAR_IMAGENS, f"processado_{imagem_nome}"
        )
        cv2.imwrite(caminho_salvar, imagem)
        # print(f"Imagem processada salva em: {caminho_salvar}")

    limpar_pasta_temp(CAMINHO_TEMP)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        # normalizar_nomes(PATH_IMAGES["teste"])
        # normalizar_referencias(PATH_IMAGES["referencia"])
        # extrair_faces(PATH_IMAGES["recortar"], PATH_IMAGES["faces"])
        processar_deteccao_face()
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")
        sys.exit(0)
