# Objetivo del Sistema: ML Predictivo de Alto Rendimiento
1. **Modelo**: Cambiar scripts estáticos por un modelo XGBoost o Random Forest que aprenda de datos históricos.
2. **Gestión de Riesgo**: Implementar Criterio de Kelly Fraccional para maximizar el crecimiento del bankroll y minimizar el riesgo de ruina.
3. **Feedback Loop**: El sistema debe guardar cada predicción y, cuando el resultado real ocurra, comparar y re-entrenar el modelo automáticamente.
4. **Features**: Calcular variables como "Momentum", "Efectividad Ofensiva/Defensiva" y "Value Gap" (diferencia entre probabilidad real y cuota de la casa).
