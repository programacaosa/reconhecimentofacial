import streamlit as st
import cv2
import numpy as np

# Função para capturar uma imagem única e reconhecer o rosto
def capture_face():
    # Iniciar captura de vídeo
    video_capture = cv2.VideoCapture(0)
    
    # Carregar o classificador Haar para detectar rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Exibir o título e a instrução
    st.title("Detecção Facial com OpenCV e Streamlit")
    st.write("Por favor, olhe para a câmera e fique parado. O sistema capturará uma foto quando detectar seu rosto.")

    # Flag para indicar quando a foto foi tirada
    face_detected = False
    captured_image = None

    while not face_detected:
        # Captura um frame da webcam
        ret, frame = video_capture.read()

        # Converter a imagem para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Captura a imagem do rosto
            captured_image = frame[y:y+h, x:x+w]
            face_detected = True  # Rosto detectado, sai do loop

        # Exibir o frame da webcam em tempo real com o retângulo ao redor do rosto
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Liberar a captura de vídeo
    video_capture.release()

    return captured_image

# Função para exibir a foto tirada após a detecção e o link
def display_captured_face_and_link(captured_image):
    # Exibir uma nova tela com a foto do rosto
    st.title("Rosto Detectado")
    st.image(captured_image, channels="RGB", use_column_width=True)

    # Exibir o link após a detecção
    st.write("O rosto foi detectado! Clique no link abaixo:")
    
    # Aqui você pode colocar o link para qualquer página que desejar
    link = "[Clique aqui para ver mais](https://www.exemplo.com)"
    st.markdown(link, unsafe_allow_html=True)

# Chamar a função principal para capturar o rosto e exibir a imagem
if __name__ == "__main__":
    captured_image = capture_face()  # Captura da imagem com detecção
    display_captured_face_and_link(captured_image)  # Exibe a imagem capturada e o link
