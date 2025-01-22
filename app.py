import os
import streamlit as st
from faster_whisper import WhisperModel
import tempfile
from pydub import AudioSegment
import warnings

# Configuraciones iniciales
warnings.filterwarnings('ignore', category=SyntaxWarning)
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"

# Configuración de la página
st.set_page_config(
    page_title="Transcriptor de Video",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Funciones auxiliares
def format_time(seconds):
    """Convierte segundos a formato HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@st.cache_data
def load_whisper_model(model_size):
    """Carga el modelo Whisper con caché"""
    return WhisperModel(model_size)

def extract_audio(video_file):
    """Extrae el audio de un archivo de video"""
    try:
        # Crear directorio temporal si no existe
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "temp_video.mp4")
        audio_path = os.path.join(temp_dir, "temp_audio.wav")

        # Guardar video
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Convertir a audio
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(audio_path, format='wav')

        return audio_path, temp_dir
    except Exception as e:
        st.error(f"Error al procesar el audio: {str(e)}")
        return None, None

def cleanup_files(temp_dir):
    """Limpia archivos temporales"""
    if temp_dir and os.path.exists(temp_dir):
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except Exception as e:
            st.warning(f"No se pudieron eliminar algunos archivos temporales: {str(e)}")

def main():
    st.title("Transcriptor de Video a Texto 🎥➡️📝")
    
    # Contenedor para configuraciones
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            model_size = st.selectbox(
                "Modelo de transcripción",
                ["tiny", "base", "small", "medium", "large"],
                help="Modelos más grandes son más precisos pero más lentos"
            )
        
        with col2:
            language = st.selectbox(
                "Idioma del video",
                ["Autodetectar", "Español", "English"],
                help="Selecciona el idioma del video para mejor precisión"
            )

    # Mapeo de idiomas
    language_map = {
        "Autodetectar": None,
        "Español": "es",
        "English": "en"
    }

    # Área de carga de archivo
    st.markdown("### Subir Video")
    video_file = st.file_uploader(
        "Formatos soportados: MP4, AVI, MOV, MKV",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Selecciona un archivo de video para transcribir"
    )

    # Estado de la aplicación
    if 'transcription_done' not in st.session_state:
        st.session_state.transcription_done = False
    
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None

    # Procesar video
    if video_file and st.button("Transcribir Video", help="Iniciar proceso de transcripción"):
        try:
            # Limpiar estado anterior
            if st.session_state.temp_dir:
                cleanup_files(st.session_state.temp_dir)
            
            with st.spinner("Procesando video..."):
                # Extraer audio
                audio_path, temp_dir = extract_audio(video_file)
                st.session_state.temp_dir = temp_dir

                if audio_path:
                    # Cargar modelo
                    model = load_whisper_model(model_size)
                    
                    # Transcribir
                    segments, _ = model.transcribe(
                        audio_path,
                        language=language_map[language]
                    )

                    # Formatear resultados
                    transcript = "\n".join([
                        f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}"
                        for segment in segments
                    ])

                    # Guardar en session state
                    st.session_state.transcript = transcript
                    st.session_state.transcription_done = True

                    # Limpiar archivos
                    cleanup_files(temp_dir)
                    
        except Exception as e:
            st.error(f"Error durante la transcripción: {str(e)}")
            if st.session_state.temp_dir:
                cleanup_files(st.session_state.temp_dir)

    # Mostrar resultados
    if st.session_state.get('transcription_done', False):
        st.markdown("### Resultado de la Transcripción")
        st.text_area(
            "Texto transcrito:",
            st.session_state.transcript,
            height=400,
            help="Puedes copiar el texto o descargarlo usando el botón de abajo"
        )
        
        # Botón de descarga
        st.download_button(
            label="📥 Descargar Transcripción",
            data=st.session_state.transcript,
            file_name="transcripcion.txt",
            mime="text/plain",
            help="Descargar el texto transcrito en formato TXT"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Desarrollado con ❤️ usando Streamlit y Whisper</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
