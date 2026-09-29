import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import edge_tts
import asyncio
import speech_recognition as sr
from groq import Groq

# --- CONFIGURAÇÃO DA IA (GROQ) ---
CHAVE_GROQ = "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"
client = Groq(api_key=CHAVE_GROQ)


# --- DESCOBERTA AUTOMÁTICA BLINDADA E FILTRADA ---
@st.cache_data
def obter_lista_modelos():
    try:
        temp_client = Groq(api_key=CHAVE_GROQ)
        modelos = temp_client.models.list().data
        palavras_bloqueadas = [
            "whisper",
            "guard",
            "canopylabs",
            "vision",
            "embed",
            "llava",
        ]
        return [
            m.id
            for m in modelos
            if not any(p in m.id.lower() for p in palavras_bloqueadas)
        ]
    except Exception:
        return ["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"]


LISTA_MODELOS = obter_lista_modelos()

st.set_page_config(page_title="IdleChatter", page_icon="🗣️", layout="wide")

# --- INÍCIO DO SISTEMA DE LOGIN ---
if "logado" not in st.session_state:
    st.session_state["logado"] = False

if not st.session_state["logado"]:

    # --- TRUQUE DO ARREDONDAMENTO DA IMAGEM ---
    st.markdown(
        """
    <style>
        [data-testid="stImage"] img {
            border-radius: 30px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # ------------------------------------------

    # Mostra o ecrã de login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Sub-colunas para manter a imagem e os formulários pequenos e no centro
        col_img1, col_img2, col_img3 = st.columns([1, 1, 1])

        # AGORA TUDO FICA DENTRO DA COLUNA DO MEIO (col_img2)
        with col_img2:
            st.image("logo.png", use_container_width=True)

            usuario = st.text_input("Login")
            senha = st.text_input("Senha", type="password")

            if st.button("Entrar", use_container_width=True):
                if usuario == "Kassio" and senha == "chatter2026":
                    st.session_state["logado"] = True
                    st.session_state["nome_usuario"] = (
                        usuario.capitalize()
                    )  # Guarda "Cassio" na memória
                    st.rerun()
                else:
                    st.error("Utilizador ou palavra-passe incorretos.")

    st.stop()  # Isto impede o Python de ler o código abaixo se não houver login.
# --- FIM DO SISTEMA DE LOGIN ---

# --- CONTROLE DO MICROFONE (Para forçar ele a limpar) ---
if "chave_audio" not in st.session_state:
    st.session_state["chave_audio"] = 0
    if "ultimo_audio_id" not in st.session_state:
        st.session_state["ultimo_audio_id"] = None

# --- MEMÓRIA DO CHAT AVANÇADA E SAUDAÇÃO DO CHATTER-B ---
if "mensagens" not in st.session_state or len(st.session_state["mensagens"]) == 0:
    nome = st.session_state.get("nome_usuario", "visitante")
    msg_inicial = f"Olá, {nome}. Sou o Chatter-B e estou aqui pra gente conversar em Inglês. Eu vou te responder em Inglês, avaliar a sua pronúncia e te dar dicas de vocabulário e gramática. Vamos lá!"
    # A primeira mensagem do tutor já entra na memória!
    st.session_state["mensagens"] = [{"role": "assistant", "content": msg_inicial}]


# --- RENDERIZA O HISTÓRICO DE CONVERSA (LINHA APÓS LINHA) ---
if st.session_state.mensagens:
    for msg in st.session_state.mensagens:
        icone = "Chart-B.png" if msg["role"] == "assistant" else "kassio.png"
        with st.chat_message(msg["role"], avatar=icone):
            st.write(msg["content"])

            # Se for a vez do utilizador, mostra as métricas e áudios
            if msg["role"] == "user" and "score" in msg:
                st.metric(label="Avaliação de Pronúncia", value=f"{msg['score']:.1f}%")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Sua fala:")
                    st.audio(msg["user_audio"], format="audio/wav")
                with col2:
                    st.write("Referência de fala:")
                    st.audio(msg["native_audio"], format="audio/mp3")

            # Se for a vez do tutor, mostra o áudio, feedback e tradução
            if msg["role"] == "assistant":
                if "audio" in msg:
                    st.audio(msg["audio"], format="audio/mp3")
                if msg.get("feedback"):
                    with st.expander("📝 Feedback Chatter-B"):
                        st.write(msg["feedback"])
                if msg.get("traducao"):
                    with st.expander("🇧🇷 Ver Tradução"):
                        st.write(msg["traducao"])
else:
    st.info(
        "👋 O chat está vazio. Use o gravador no fundo da página e diga a sua primeira frase em inglês para iniciar!"
    )

# --- 1. O SEGREDO: CRIAR O CONTENTOR AQUI (Acima do microfone) ---
novo_turno_container = st.container()

st.divider()  # Adiciona uma linha discreta para separar a conversa do gravador

# --- 2. MICROFONE E BOTÃO DE LIMPAR (Ficam presos no fundo e à esquerda) ---

# Cria duas colunas (metade/metade do ecrã). O que estiver na col1 fica à esquerda.
col1, col2 = st.columns(2)

with col1:
    audio_file = st.audio_input(
        "Aperte o microfone e diga alguma coisa:",
        key=f"mic_{st.session_state['chave_audio']}",
    )

    # Removido o 'use_container_width=True' (ou mantido apenas para alinhar com o microfone)
    if st.button("🔄 Nova Conversa"):
        st.session_state["mensagens"] = []
        st.session_state["chave_audio"] += 1
        st.rerun()


# --- FUNÇÕES DO MOTOR DE ÁUDIO ---
def comparar_audios(caminho_nativo, caminho_usuario):
    y1, sr1 = librosa.load(caminho_nativo, duration=10)
    y2, sr2 = librosa.load(caminho_usuario, duration=10)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    D, wp = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric="cosine")
    distancia_media = D[-1, -1] / len(wp)
    score = max(0.0, min(100.0, 100 - (distancia_media * 200)))
    return score


def gerar_audio_nativo(texto):
    # Aqui você escolhe o "ator" da voz.
    voz = "en-US-AnaNeural"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    asyncio.run(edge_tts.Communicate(texto, voz, pitch="-50Hz").save(tmp.name))

    # Executa a geração neural da Microsoft e guarda no arquivo temporário
    asyncio.run(edge_tts.Communicate(texto, voz).save(tmp.name))

    return tmp.name


def transcrever_audio(caminho_audio):
    r = sr.Recognizer()
    with sr.AudioFile(caminho_audio) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language="en-US")
    except (sr.UnknownValueError, sr.RequestError):
        return None


# --- PROCESSAMENTO DO ÁUDIO NOVO ---
if audio_file is not None:
    id_audio_atual = f"{audio_file.name}_{audio_file.size}"

    if id_audio_atual != st.session_state.ultimo_audio_id:
        # Direciona todo o desenho da interface para o contentor acima do microfone
        with novo_turno_container:
            audio_bytes = audio_file.read()
            status_text = st.empty()
            path_user = None
            path_nativo = None
            path_audio_ia = None

    try:
        status_text.info("A ouvir a sua pronúncia...")
        tmp_user = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp_user.write(audio_bytes)
        tmp_user.close()
        path_user = tmp_user.name

        texto_reconhecido = transcrever_audio(path_user)

        if texto_reconhecido:
            st.session_state.ultimo_audio_id = id_audio_atual

            # 1. Processa e guarda a ação do utilizador
            status_text.info("A comparar com o sotaque nativo...")
            path_native = gerar_audio_nativo(texto_reconhecido)
            score = comparar_audios(path_native, path_user)
            with open(path_native, "rb") as f:
                native_audio_bytes = f.read()

            st.session_state.mensagens.append(
                {
                    "role": "user",
                    "content": texto_reconhecido,
                    "score": score,
                    "user_audio": audio_bytes,
                    "native_audio": native_audio_bytes,
                }
            )

            # Renderiza imediatamente a fala do utilizador na tela
            with st.chat_message("user"):
                st.write(texto_reconhecido)
                st.metric(label="Avaliação de Pronúncia", value=f"{score:.1f}%")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Sua fala:")
                    st.audio(audio_bytes, format="audio/wav")
                with col2:
                    st.write("Referência de fala:")
                    st.audio(native_audio_bytes, format="audio/mp3")

            # 2. Chama a IA para dar continuidade ao contexto
            status_text.info("O tutor está a formular a resposta e a gerar voz...")

            historico_texto = ""
            for msg in st.session_state.mensagens[:-1]:
                prefixo = "Você" if msg["role"] == "user" else "Tutor"
                historico_texto += f"{prefixo}: {msg['content']}\n"

            # PROMPT AGRESSIVO PARA OBRIGAR A TRADUÇÃO
            prompt_ia = f"""
            Você é o Chatter-B, um tutor amigável de conversação em inglês. 
            Sempre que o estudante perguntar o seu nome ou quem você é, responda de forma natural em inglês que o seu nome é Chatter-B.
    Histórico da conversa:
    {historico_texto}

    O estudante acabou de dizer: "{texto_reconhecido}"

    REGRA MÁXIMA: Você DEVE estruturar a sua resposta EXATAMENTE com os três marcadores abaixo (incluindo os colchetes). É estritamente proibido não usar os marcadores.

    [INGLES]
    Escreva aqui APENAS a sua resposta natural em INGLÊS, mantendo o diálogo vivo.

    [FEEDBACK]
    Escreva aqui em PORTUGUÊS DO BRASIL a sua avaliação pedagógica. Diga se a construção gramatical foi boa e sugira vocabulário.

    [TRADUCAO]
    Escreva aqui a TRADUÇÃO para o português do Brasil da frase do estudante e da sua resposta. Abaixo um do outro com o Marcador Resposta do Estudante e Resposta do Tutor.
    """

            texto_tutor = None
            for modelo_teste in LISTA_MODELOS:
                try:
                    resposta = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_ia}],
                        model=modelo_teste,
                    )
                    texto_tutor = resposta.choices[0].message.content
                    break
                except Exception:
                    continue

            if texto_tutor:
                # --- 1. A TESOURA (Corta o texto nos marcadores) ---
                ingles = texto_tutor
                feedback = ""
                traducao = ""

                if "[FEEDBACK]" in texto_tutor:
                    partes = texto_tutor.split("[FEEDBACK]")
                    ingles = partes[0].replace("[INGLES]", "").strip()

                    if "[TRADUCAO]" in partes[1]:
                        sub_partes = partes[1].split("[TRADUCAO]")
                        feedback = sub_partes[0].strip()
                        traducao = sub_partes[1].strip()
                    else:
                        feedback = partes[1].strip()

                # 2. Gera o áudio APENAS da parte em inglês limpa
                path_audio_ia = gerar_audio_nativo(ingles)
                with open(path_audio_ia, "rb") as f:
                    audio_ia_bytes = f.read()

                    # 3. Guarda na memória do chat com as gavetas separadas
                    st.session_state.mensagens.append(
                        {
                            "role": "assistant",
                            "content": ingles,
                            "feedback": feedback,
                            "traducao": traducao,
                            "audio": audio_ia_bytes,
                        }
                    )
                    st.session_state["chave_audio"] += 1  # Limpa o microfone
                    st.rerun()  # Recarrega a página instantaneamente

            else:
                st.error("Falha na conexão com a IA.")
                status_text.empty()
        else:
            status_text.empty()
            st.error(
                "Não conseguimos entender o áudio. Tente falar de forma mais clara."
            )

    except Exception as e:
        status_text.empty()
        st.error(f"Ocorreu um erro: {e}")

    finally:
        try:
            if path_user and os.path.exists(path_user):
                os.remove(path_user)
            if path_native and os.path.exists(path_native):
                os.remove(path_native)
            if path_audio_ia and os.path.exists(path_audio_ia):
                os.remove(path_audio_ia)
        except Exception:
            pass
