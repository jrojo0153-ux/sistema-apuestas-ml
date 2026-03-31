# 🚀 Sistema de Apuestas con Inteligencia Artificial (ML)

Este es un bot de **Value Betting** automatizado que utiliza Machine Learning para identificar oportunidades de valor en el fútbol. El sistema analiza partidos en tiempo real, calcula probabilidades mediante un modelo de **Random Forest** y gestiona el bankroll utilizando el **Criterio de Kelly**.

## 🛠️ Características Principales
- **IA Predictiva:** Modelo entrenado con Scikit-learn que analiza la relación entre cuotas y resultados históricos.
- **Value Betting:** Identifica cuando la probabilidad de la IA es mayor a la implícita en la cuota de la casa de apuestas.
- **Gestión de Bankroll:** Implementación de Kelly fraccionado para maximizar el crecimiento y minimizar el riesgo.
- **Automatización:** Integrado con **GitHub Actions** para ejecutarse cada 6 horas automáticamente.
- **Alertas en Tiempo Real:** Notificaciones directas a **Telegram** cuando se detecta una apuesta de valor.

## 📂 Estructura del Proyecto
```text
├── .github/workflows/  # Automatización (CI/CD)
├── core/               # Lógica de valor y EV (Expected Value)
├── data/               # Historico.csv y scripts de extracción
├── ml/                 # Entrenamiento y carga del modelo IA
├── models/             # Almacenamiento del modelo generado (.pkl)
├── portfolio/          # Gestión de bankroll y stakes
├── config.py           # Parámetros de configuración del bot
└── main.py             # Punto de entrada principal

🚀 Instalación y Configuración
​1. Requisitos
​Python 3.10+
​Pip (instalar dependencias: pip install -r requirements.txt)
​2. Variables de Entorno (GitHub Secrets)
​Para que el bot funcione en la nube, debes configurar los siguientes Secrets en tu repositorio:
​API_FOOTBALL_KEY: Tu API Key de Football API (API-Sports).
​TELEGRAM_BOT_TOKEN: Token de tu bot creado con @BotFather.
​TELEGRAM_CHAT_ID: Tu ID de chat de Telegram.
​3. Funcionamiento
​El bot sigue este flujo lógico:
​Entrenamiento: Carga data/Historico.csv y genera un modelo de clasificación.
​Análisis: Obtiene los partidos del día vía API.
​Predicción: La IA calcula la probabilidad de victoria local.
​Evaluación: Si hay valor (Prob > 1/Cuota), calcula el stake óptimo.
​Notificación: Envía un mensaje detallado a Telegram.
​📊 Modelo de IA
​El corazón del sistema es un Random Forest Classifier que busca patrones de ineficiencia en las cuotas. Actualmente el sistema se auto-optimiza cada vez que se actualiza el dataset histórico.
​Disclaimer: Este software es una herramienta de análisis estadístico. El uso de apuestas deportivas conlleva riesgos financieros. Juega con responsabilidad.
