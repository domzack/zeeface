# reconhecimento_deepface.py

import cv2
from mtcnn import MTCNN  # Altere para o módulo original
from deepface import DeepFace
import os
import sys


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


def main():
    # Configurações
    CAMINHO_IMAGENS = "./imagens"  # Diretório atual
    CAMINHO_REFERENCIAS = "./banco_referencias"
    CAMINHO_SALVAR_IMAGENS = "./imagens_processadas"
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

        print(f"Processando: {imagem_nome}")
        caminho_imagem = os.path.join(CAMINHO_IMAGENS, imagem_nome)
        imagem = cv2.imread(caminho_imagem)

        faces = detectar_faces(imagem)
        if not faces:
            print(f"Nenhuma face detectada em {imagem_nome}.")
            continue

        for i, face in enumerate(faces):
            x, y, w, h = face["box"]
            face_crop = imagem[y : y + h, x : x + w]

            # Salva temporariamente a face para reconhecimento na pasta temp
            nome_face = os.path.join(CAMINHO_TEMP, f"temp_face_{i+1}.jpg")
            cv2.imwrite(nome_face, face_crop)

            identidade = reconhecer_face(nome_face, CAMINHO_REFERENCIAS, MODELOS)

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

        # Salva a imagem processada no diretório
        caminho_salvar = os.path.join(
            CAMINHO_SALVAR_IMAGENS, f"processado_{imagem_nome}"
        )
        cv2.imwrite(caminho_salvar, imagem)
        print(f"Imagem processada salva em: {caminho_salvar}")

    limpar_pasta_temp(CAMINHO_TEMP)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário.")
        sys.exit(0)

# Comando para conectar ao repositório remoto
# Execute no terminal:
# git remote add origin https://github.com/domzack/zeeface.git
# git push -u origin main
