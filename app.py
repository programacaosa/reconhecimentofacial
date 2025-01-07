import streamlit as st
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Função para exibir o HTML e JavaScript para capturar a imagem da câmera do navegador
def capture_face_from_browser():
    # HTML e JavaScript para acessar a câmera do navegador
    st.markdown("""
    <script>
        // Cria o vídeo no HTML
        let video = document.createElement('video');
        video.width = 640;
        video.height = 480;
        document.body.appendChild(video);

        // Acessa a câmera do navegador
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function(error) {
            alert("Erro ao acessar a câmera: " + error);
        });

        // Função para capturar a imagem
        function captureImage() {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const context = canvas.getContext('2d');
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Converte a imagem para base64 e envia para o Streamlit
            const imageData = canvas.toDataURL('image/png');
            const image = imageData.split(',')[1]; // Remove o prefixo 'data:image/png;base64,'
            
            // Envia a imagem para o Streamlit usando o Python
            window.parent.postMessage({ image: image }, "*");
        }

        // Captura a imagem quando o usuário clicar no botão
        document.body.innerHTML += '<button onclick="captureImage()">Capturar Foto</button>';
    </script>
    """, unsafe_allow_html=True)

# Função para processar a imagem com OpenCV
def process_image(image_data):
    # Decodificando a imagem base64
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    open_cv_image = np.array(image)

    # Convertendo a imagem para o formato BGR (para o OpenCV)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Detecção de rosto com OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Desenhando retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Convertendo de volta para imagem RGB para exibição no Streamlit
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(open_cv_image)

# Exibir a captura da imagem do navegador e processá-la
if __name__ == "__main__":
    st.title("Detecção Facial com OpenCV e Streamlit")

    # Captura da imagem através do navegador
    capture_face_from_browser()

    # Esperar a imagem ser enviada
    image_data = st.experimental_get_query_params().get('image', [None])[0]

    if image_data:
        processed_image = process_image(image_data)
        st.image(processed_image, caption="Imagem com Rosto Detectado", use_column_width=True)
