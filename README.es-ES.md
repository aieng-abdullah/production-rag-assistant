

<div align="center">

<img width="300" height="300" alt="artificial-intelligence" src="https://github.com/user-attachments/assets/b92417d0-a09f-4353-883b-d6f545e727e8" />

# Asistente de Investigación RAG para Producción

[![CI](https://github.com/aieng-abdullah/production-rag-assistant/actions/workflows/eval.yml/badge.svg)](https://github.com/aieng-abdullah/production-rag-assistant/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green)](https://langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-live-red?logo=streamlit)](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Anthropic%20%7C%20OpenAI-orange)](https://groq.com)
[![Langfuse](https://img.shields.io/badge/Observability-Langfuse-purple)](https://langfuse.com)
[![Ragas](https://img.shields.io/badge/Evaluated-Ragas-blue)](https://ragas.io)

### Sistema de Generación Aumentada por Recuperación (RAG) de grado producción para artículos de investigación

Recuperación Híbrida • Reclasificación con Cross-Encoder • Validación de Citas • Evaluación Automatizada

---

## Enlace en Vivo⚡

[Probar la aplicación en vivo](https://appuction-rag-assistant-hlmgqebzhhynbgpbnnekqw.streamlit.app/)

Sube artículos de investigación en PDF y haz preguntas con citas fundamentadas y referencias de página.

</div>

---

# El Problema que Resuelve

Los investigadores académicos, ingenieros y estudiantes leen decenas de artículos
para encontrar respuestas específicas — y aún así quedan inseguros sobre si lo entendieron
correctamente.

Preguntar a un LLM estándar es peor. Da respuestas seguras y bien redactadas
que pueden no tener nada que ver con lo que el artículo dice realmente.
No puedes citar una alucinación.

**Este sistema te permite consultar artículos de investigación y confiar en las respuestas.**

Sube uno o varios PDFs. Haz una pregunta específica. Cada respuesta
incluye citas exactas `[SOURCE N]` y referencias de página, fundamentadas
en lo que el documento contiene realmente, no en lo que el modelo asume.

Si la fuente no respalda la respuesta, esta se rechaza antes de llegar a ti.

---

### Para quién está diseñado

| Si eres... | Esto resuelve... |
|---|---|
| Un investigador | Síntesis entre artículos sin tener que hacer un recorrido manual |
| Un ingeniero | Extracción de detalles de implementación desde artículos técnicos |
| Un estudiante | Respuestas citables que realmente puedes referenciar en tus escritos |

---

# Arquitectura

## Pipeline de Procesamiento de Documentos

```text
PDF Upload
    ↓
PyMuPDF Parser
    • Page-aware text extraction

    ↓
RecursiveCharacterTextSplitter
    • 350 token chunks
    • 75 overlap

    ↓
HuggingFace Embeddings
    • sentence-transformers/all-MiniLM-L6-v2

    ↓
ChromaDB
    • Vector storage with cosine similarity
```

---

## Pipeline de Recuperación y Generación

```text
User Query
     ├── BM25 Search (Top 20)
     ├── Vector Search (Top 20)
     ↓
Reciprocal Rank Fusion (RRF)
     ↓
Cross-Encoder Reranker
     • ms-marco-MiniLM-L-6-v2

     ↓
Top 5 Chunks
     ↓
Citation Prompt Builder
     ↓
LLM Provider Chain (with retry + exponential backoff)
     ├── Groq (Llama 3.3 70B) — free
     ├── Anthropic (Claude) — optional, user-provided key
     └── OpenAI (GPT-4o) — optional, user-provided key
     ↓
Pydantic Citation Validator
     ↓
Final Response with [SOURCE N] Citations
```

---

# Decisiones Técnicas Clave

<details>
<summary><b>¿Por qué recuperación híbrida en lugar de solo vectorial?</b></summary>

<br>

BM25 sobresale en la coincidencia exacta de palabras clave.

Esto es crítico para la terminología técnica de investigación como:

- "scaled dot-product attention"
- "BLEU score"
- "adaptadores LoRA"

La recuperación vectorial maneja la similitud semántica.

La Fusión de Rangos Recíprocos (RRF) combina ambos sistemas de recuperación sin requerir normalización de puntuaciones entre los métodos.

</details>

---

<details>
<summary><b>¿Por qué usar reclasificación con cross-encoder?</b></summary>

<br>

Los bi-encoders incrustan consultas y fragmentos de forma independiente.

Los cross-encoders evalúan la consulta y el fragmento juntos, produciendo una puntuación de relevancia significativamente más precisa.

Ejecutar la inferencia del cross-encoder en todos los fragmentos sería computacionalmente costoso.

En su lugar, la reclasificación se aplica solo a los principales candidatos de recuperación después de la fusión RRF.

</details>

---

<details>
<summary><b>¿Por qué forzar las citas con validación Pydantic?</b></summary>

<br>

Las instrucciones del prompt por sí solas son poco confiables.

El validador falla inmediatamente si la respuesta no contiene patrones válidos de `[SOURCE N]`.

Esto fuerza respuestas fundamentadas en lugar de depender enteramente del cumplimiento del prompt.

</details>

---

<details>
<summary><b>¿Por qué observabilidad con Langfuse?</b></summary>

<br>

Los sistemas de IA de producción no pueden depurarse efectivamente usando solo registros.

Langfuse rastrea:

- Latencia de recuperación
- Construcción del prompt
- Uso de tokens
- Salidas del LLM
- Validación de citas

El rastreo identificó que el reclasificador cross-encoder representa aproximadamente el 72% de la latencia total.

</details>

---

# Resultados de Evaluación

Evaluado en 5 pares de pregunta-respuesta del artículo *Attention Is All You Need* usando métricas Ragas con el LLM de Groq como juez.

| Métrica | Puntuación | Umbral | Estado |
|---|---|---|---|
| Fidelidad (Faithfulness) | **0.83** | 0.75 | ![PASS](https://img.shields.io/badge/PASS-success) |
| Relevancia de la Respuesta | **0.90** | 0.75 | ![PASS](https://img.shields.io/badge/PASS-success) |
| Recordación del Contexto (Recall) | **1.00** | 0.70 | ![PASS](https://img.shields.io/badge/PASS-success) |

---

### Fidelidad — 0.83

El 83% de las afirmaciones generadas están fundamentadas en el contexto recuperado.

El sistema no está fabricando respuestas sin respaldo a una tasa significativa.

---

### Relevancia de la Respuesta — 0.90

Las respuestas abordan directamente la consulta del usuario con una salida irrelevante mínima.

Métrica de evaluación con mayor puntuación.

---

### Recordación del Contexto — 1.00

El pipeline de recuperación obtuvo exitosamente toda la información requerida para cada consulta de evaluación.

No se omitió ningún fragmento relevante.

---

### Limitación Conocida

La precisión del contexto es menor debido a la superposición de fragmentos académicos.

Optimizaciones planificadas:

- Reducir el tamaño del fragmento de 350 → 256
- Añadir filtrado de metadatos consciente de secciones

---

# Observabilidad y Monitoreo

Todas las solicitudes se rastrean de extremo a extremo usando Langfuse.

Se recopilaron 141 trazas completas de solicitudes a través del uso real.

---

### Perfil de Latencia

Basado en 141 solicitudes rastreadas.

| Métrica | Latencia | Qué Significa |
|--------|---------|---------------|
| p50 | 1.54s | La mayoría de los usuarios experimenta esto |
| p90 | 7.32s | 1 de cada 10 usuarios espera este tiempo |
| p95 | 11.09s | 1 de cada 20 usuarios espera este tiempo |
| p99 | 14.06s | Peor caso observado |

La experiencia mediana es de 1.54 segundos.

La varianza es impulsada por el reclasificador cross-encoder, no por el LLM.

---

### Desglose de Latencia por Componente

| Componente | p50 | p90 | p95 | p99 | Función |
|-----------|-----|-----|-----|-----|------|
| Solicitud Completa | 1.54s | 7.32s | 11.09s | 14.06s | De extremo a extremo |
| Recuperación | 0.77s | 6.11s | 10.01s | 12.19s | BM25 + vector + RRF |
| Reclasificación | 0.74s | 6.03s | 9.23s | 11.68s | Puntuación cross-encoder |
| ChatGroq (LLM) | 0.57s | 0.93s | 1.19s | 2.52s | Generación |
| Búsqueda Vectorial | 0.03s | 0.06s | 0.22s | 0.38s | Consulta de incrustaciones |

---

### Hallazgo Clave: El LLM No es el Cuello de Botella

Suposición común: La generación del LLM impulsa la latencia.

Lo que muestran los datos: El LLM contribuye solo con 0.57s en la mediana.

El reclasificador cross-encoder es el cuello de botella real.

| Componente | p50 | p95 | Ratio de Varianza |
|-----------|-----|-----|----------------|
| Reclasificación | 0.74s | 9.23s | 12x |
| Recuperación | 0.77s | 10.01s | 13x |
| ChatGroq | 0.57s | 1.19s | 2x |
| Búsqueda Vectorial | 0.03s | 0.22s | 7x |

La inferencia de Groq es rápida y estable: 2x de varianza entre p50 y p95.

El reclasificador muestra una varianza de 12x porque el cross-encoder puntúa cada par (consulta, fragmento) individualmente en la CPU.

La latencia escala con la densidad del documento y la longitud de la consulta.

Este hallazgo habría sido invisible sin instrumentación.

Optimizar el LLM — el objetivo intuitivo — habría tenido un impacto cercano a cero.

---

### Por qué el Reclasificador Tiene Alta Varianza

El reclasificador cross-encoder puntúa cada par (consulta, fragmento) individualmente.

Con 20 fragmentos candidatos por consulta, el cómputo total escala con:

- Longitud en tokens de la consulta
- Longitud en tokens del fragmento
- Número de candidatos

Consulta corta + PDF disperso  → ~0.74s

Consulta larga + PDF denso    → ~9.23s

Esta es una operación limitada por la CPU sin optimización de lotes (batching) en la implementación actual.

---

### Uso del Modelo

| Modelo | Tokens Usados | Propósito |
|-------|-------------|---------|
| llama-3.3-70b-versatile | 96,490 | Generación principal |

Tokens promedio por traza: ~684 tokens

---

### Estructura de la Traza

Cada solicitud se divide en span:

| Span | Qué Rastrea |
|------|---------------|
| recuperación | Latencia de BM25 + vector + RRF + reclasificación |
| construcción-prompt | Longitud del prompt y tiempo de construcción |
| llamada-llm | Uso de tokens, modelo, tiempo de respuesta |
| validación-citas | Estado de éxito/fallo |

Cuello de botella identificado a través de trazas de Langfuse: sin conjeturas.

---

# Conmutación por Error (Failover) de Proveedor LLM

El sistema admite **3 proveedores de LLM** con reintentos automáticos, backoff exponencial y conmutación por error:

| Proveedor | Modelo | Costo | Configuración |
|----------|-------|------|-------|
| **Groq** | Llama 3.3 70B Versatile | Gratis | Establecer `GROQ_API_KEY` en `.env` |
| **Anthropic** | Claude Sonnet 4 | Pago por uso | Añadir clave por la barra lateral o establecer `ANTHROPIC_API_KEY` |
| **OpenAI** | GPT-4o | Pago por uso | Añadir clave por la barra lateral o establecer `OPENAI_API_KEY` |

### Cómo Funciona

1. **Reintento**: Cada proveedor obtiene hasta 3 intentos con backoff exponencial (1s → 2s → 4s)
2. **Conmutación por error**: Si un proveedor falla después de los reintentos, se intenta con el siguiente en la cadena
3. **Orden de la cadena**: Groq → Anthropic → OpenAI (solo se incluyen proveedores con claves API)

### Añadir Tu Propio Proveedor

No se necesitan cambios en el código. Dos opciones:

**Opción A — Interfaz de Barra Lateral (recomendada):**
Abre la aplicación → expande "Proveedores LLM" en la barra lateral → ingresa tu clave API y nombre del modelo.

**Opción B — Variables de entorno:**
```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
```

Si solo se configura Groq (predeterminado), el sistema funciona exactamente como antes: solo con resiliencia de reintento.

---

# Pila Tecnológica

| Capa | Tecnología |
|---|---|
| Análisis de PDF | PyMuPDF |
| Fragmentación | LangChain RecursiveCharacterTextSplitter |
| Incrustaciones | sentence-transformers/all-MiniLM-L6-v2 |
| Base de Datos Vectorial | ChromaDB |
| Recuperación Dispersa | BM25Retriever |
| Reclasificador | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq / Anthropic / OpenAI (con reintento + conmutación por error) |
| Orquestación | LangChain |
| Interfaz | Streamlit |
| Observabilidad | Langfuse |
| Evaluación | Ragas |
| CI/CD | GitHub Actions |

---

# Configuración Local

<details>
<summary><b>Instrucciones de Configuración</b></summary>

<br>

```bash
# Clonar repositorio
git clone https://github.com/aieng-abdullah/production-rag-assistant.git

# Entrar al proyecto
cd production-rag-assistant

# Crear entorno virtual
python3 -m venv venv

# Activar entorno
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Añadir GROQ_API_KEY dentro de .env

# Iniciar aplicación
streamlit run app.py
```

</details>

---

# Ejecutar Evaluación

```bash
python3 eval/eval_runner.py
```

Los resultados de la evaluación se guardan en `results.json`.

---

# Ejecutar Pruebas

```bash
pytest tests/ -v
```

---

# Notas de Rendimiento

La latencia de extremo a extremo oscila entre 1.54s (p50) y 14.06s (p99) a través de 141 solicitudes rastreadas.

El reclasificador cross-encoder representa el 72% del tiempo total de ejecución en CPU, con una varianza de 12x entre p50 y p95.

La implementación actual prioriza la calidad de recuperación y las respuestas fundamentadas sobre la latencia pura.

---

# Autor

## Abdullah Al Arif

Ingeniero de IA Jr.

[GitHub](https://github.com/aieng-abdullah) • [LinkedIn](www.linkedin.com/in/abdullah-al-arif-8b58542a7)
