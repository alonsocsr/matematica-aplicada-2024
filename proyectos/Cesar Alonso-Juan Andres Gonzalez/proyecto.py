import pandas as pd
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import skfuzzy as fuzz
import time

# Modulo 1: Preprocesamiento del dataset
data = pd.read_csv("/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/train_data.csv",
                   encoding='ISO-8859-1')

# Asegurarse de que el lexicon esté disponible
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

inicio_tiempo = time.time()
# Preprocesamiento de texto
def decontracted(phrase):
    # Expansión de contracciones y eliminación de elementos no deseados
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", "can not", phrase)
    phrase = re.sub(r"@", "", phrase)
    phrase = re.sub(r"http\S+", "", phrase)
    phrase = re.sub(r"#", "", phrase)
    phrase = re.sub(r" t ", " not ", phrase)
    phrase = re.sub(r" re ", " are ", phrase)
    phrase = re.sub(r" s ", " is ", phrase)
    phrase = re.sub(r" d ", " would ", phrase)
    phrase = re.sub(r" ll ", " will ", phrase)
    phrase = re.sub(r" ve ", " have ", phrase)
    phrase = re.sub(r" m ", " am ", phrase)
    return phrase


# Universos de variables para fuzificación
x_p = np.arange(0, 1, 0.1)
x_n = np.arange(0, 1, 0.1)
x_op = np.arange(0, 10, 1)

# Funciones de membresía
p_lo = fuzz.trimf(x_p, [0, 0, 0.5])
p_md = fuzz.trimf(x_p, [0, 0.5, 1])
p_hi = fuzz.trimf(x_p, [0.5, 1, 1])
n_lo = fuzz.trimf(x_n, [0, 0, 0.5])
n_md = fuzz.trimf(x_n, [0, 0.5, 1])
n_hi = fuzz.trimf(x_n, [0.5, 1, 1])
op_Neg = fuzz.trimf(x_op, [0, 0, 5])
op_Neu = fuzz.trimf(x_op, [0, 5, 10])
op_Pos = fuzz.trimf(x_op, [5, 10, 10])


# Función de benchmark
def obtener_puntaje_sentimiento(text):
    start_time = time.time()  # Inicio de tiempo de ejecución
    new_text = decontracted(text)
    ss = sia.polarity_scores(new_text)
    posscore = ss['pos']
    negscore = ss['neg']

    # Fuzificación
    p_level_lo = fuzz.interp_membership(x_p, p_lo, posscore)
    p_level_md = fuzz.interp_membership(x_p, p_md, posscore)
    p_level_hi = fuzz.interp_membership(x_p, p_hi, posscore)
    n_level_lo = fuzz.interp_membership(x_n, n_lo, negscore)
    n_level_md = fuzz.interp_membership(x_n, n_md, negscore)
    n_level_hi = fuzz.interp_membership(x_n, n_hi, negscore)

    # Reglas activas
    active_rule1 = np.fmin(p_level_lo, n_level_lo)
    active_rule2 = np.fmin(p_level_md, n_level_lo)
    active_rule3 = np.fmin(p_level_hi, n_level_lo)
    active_rule4 = np.fmin(p_level_lo, n_level_md)
    active_rule5 = np.fmin(p_level_md, n_level_md)
    active_rule6 = np.fmin(p_level_hi, n_level_md)
    active_rule7 = np.fmin(p_level_lo, n_level_hi)
    active_rule8 = np.fmin(p_level_md, n_level_hi)
    active_rule9 = np.fmin(p_level_hi, n_level_hi)

    n1 = np.fmax(active_rule4, active_rule7)
    n2 = np.fmax(n1, active_rule8)
    op_activation_lo = np.fmin(n2, op_Neg)

    neu1 = np.fmax(active_rule1, active_rule5)
    neu2 = np.fmax(neu1, active_rule9)
    op_activation_md = np.fmin(neu2, op_Neu)

    p1 = np.fmax(active_rule2, active_rule3)
    p2 = np.fmax(p1, active_rule6)
    op_activation_hi = np.fmin(p2, op_Pos)

    aggregated = np.fmax(op_activation_lo, np.fmax(op_activation_md, op_activation_hi))
    numerador = np.sum(x_op * aggregated)
    denominador = np.sum(aggregated)
    op = numerador / denominador if denominador != 0 else 0
    output = round(op, 2)

    # Clasificación de sentimiento
    if 0 < output < 3.33:
        sentiment = "Negative"
    elif 3.33 < output < 6.67:
        sentiment = "Neutral"
    elif 6.67 < output < 10:
        sentiment = "Positive"
    else:
        sentiment = "Neutral" #Valor predeterminado para evitar UnboundLocalError
    end_time = time.time()  # Fin de tiempo de ejecución
    exec_time = round(end_time - start_time, 4)  # Calcular tiempo de ejecución

    return posscore, negscore, sentiment, exec_time


# Aplicación de la función y recopilación de resultados
resultados = data['sentence'].apply(obtener_puntaje_sentimiento).tolist()
resultados_df = pd.DataFrame(resultados, columns=['TweetPos', 'TweetNeg', 'New_Sentiment', 'ExecTime'],
                             index=data.index)
data[['TweetPos', 'TweetNeg', 'New_Sentiment', 'ExecTime']] = resultados_df

# Cálculos de benchmark
total_positive = len(data[data['New_Sentiment'] == 'Positive'])
total_negative = len(data[data['New_Sentiment'] == 'Negative'])
total_neutral = len(data[data['New_Sentiment'] == 'Neutral'])
avg_exec_time = data['ExecTime'].mean()



# Guardar resultados en un nuevo archivo CSV
output_file_path = '/home/cesaralonso/fuzzy-rule-based-sentiment-analysis/analisis_resultado.csv'
data.to_csv(output_file_path, index=False)

fin_tiempo = time.time()
tiempo_ejecucion = fin_tiempo - inicio_tiempo
# Imprimir resumen de benchmarks
print("Resumen de Benchmarks:")
print(f"Total Positivos: {total_positive}")
print(f"Total Negativos: {total_negative}")
print(f"Total Neutrales: {total_neutral}")
print(f"Tiempo promedio de ejecución total: {round(avg_exec_time, 4)} segundos")
print(f"\nTiempo total de ejecución: {round(tiempo_ejecucion, 3)} segundos")

