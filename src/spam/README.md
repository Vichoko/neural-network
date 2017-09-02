# Clasificador de SMS de Spam

## Motivacion
Dado un SMS (Texto plano), lograr clasificarlo como spam (o no spam).

## Datos
Los datos (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip) corresponden a un archivo TSV (Tab Separated Values).
### Ejemplo
```
...
ham		I HAVE A DATE ON SUNDAY WITH WILL!!
spam	XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL
ham		k...i'm watching here:)
...
```

## Pre-procesamiento de Texto
1. Transformar clases "ham" y "spam" a valores 0 y 1; respectivamente.
2. Transformar SMS (texto) a una 'bag of words' (Matriz de SizeOf(DataSet) x SizeOf(Diccionario)) de minimo tamaño posible.

Esta parte del programa basta que se ejecute una vez, dado que el resultado del procesamiento queda guardado en 3 archivos temporales en el directorio del proyecto (../temp_tdidf_*). Si hay algun problema y se quiere forzar el recalculo de este proceso basta con eliminar estos 3 archivos temporales.

Para lograr el punto 2 se hicieron varios procesos.

### Quitar Stop-words
Se eliminaron de los SMS palabras que no tienen relevancia en el ingles. Por ejemplo, "a", "the", etc. 
Esto con el fin de disminuir el tamaño del diciconario.

### Proceso de "Stemming"
Reducir las palabras similares a su raiz; por ejemplo, "run", "running", "runned" a solamente "run".
Esto disminuye el tamaño edl diccionario.

### Quitar caracteres especiales y llevar todo a minusculas
Esto con el fin de disminuir el tamaño del diccionario. Suponiendo que las letras tienen mayor semántica, en una palabra, que los numeros y símbolos especiales. Y que las palabras en minusculas y mayusculas significan lo mismo.

### Proceso de calculo de frecuencias normalizadas con TF-IDF
El concepto de "Bag of Words" corresponde a almacenar, para cada texto (fila), la frecuencia de cada uno de sus terminos (columna).
Se aplica TF-IDF sobre cada tupla (SMS, termino, data-set), para calcular frecuencia de termino * frecuencia inversa de documentos, con el fin de normalizar los valores. Obteniendo valores más grandes para palabras con poca frecuencia global en los SMS, pero con valores más pequeños para palabras comunes.

## Procesamiento de datos
Al analizar el resultado del pre-procesamiento anterior, se notó un fuerte desbalance entre los SMS etiquetados como SPAM (700) y los que no (5000).
Por lo que se incurrieron en 2 técnicas de "Sampling" para transformar el dataSet de entrenamiento y prueba.

### Over-Sampling
Balancear la cantidad de datos por cada clase, a la cantidad de la clase mayoritaria.
Para la clase minoritaria, se repiten pares (data, class) elejidos aleatoriamente hasta que ambas clases tienen la misma cantidad de datos.

### Sub-Sampling
Balancear la cantidad de datos por cada clase, a la cantidad de la clase minoritaria.
Para la clase mayoritaria, se omiten datos para que ambas clases estén igualmente distribuidas en el data-set.

## Entrenamiento de red neuronal
Se probaron diversas configuraciones, la que se obtuvo mejor resultados consistió en:
* Capa de entrada de Diccionario.size() inputs y neuronas.
* Capa oculta de 20 neuronas.
* Capa de salida de 1 neurona.

El error cuadratico minimo que se consiguió fue de 240.
![error cuadratico vs epochs](https://i.imgur.com/SbL9nkL.png)*
*Cuando vale 0, es porque dejó de aprender dado que se estabilizó el error.
C
## Observaciones y resultados
* Se notó que a partir de la 5 epoch aproximadamente, el error cuadratico dejaba de variar; quedandose en un valor cercano a 240.
* El clasificador tiende a clasificar todo como no-spam (con threshold de 0.5).
* El clasificar tiende a clasificar todo como spam (con threshold de 0.25).
* El desempeño no varia sustancialmente si se normaliza el "bag of Words" previamente.
* Al variar la cantidad de neuronas por capa no varía el desempeño notablemente.

## Conclusión
Con estos datos no se puede resolver el proeblema de clasificacion de una manera precisa. Lo cual se puede deber a la baja cantidad de datos que se tienen (6000 aproximadamente). Dado que al probar el desempeño de la red neuronal con los test, se puede comprobar que funciona correctamente.

# Ejecuciónn
Para ejecutar el experimento se debe ubicar el archivo de datos (../SMSSpamCollection) en el directorio del proyecto.
Luego ejecutar la clase spam.MainClass.
Se debería visualizar en consola el progreso del programa (pre-procesamiento, entrenamiento y metricas).
Además quedará un archivo en el directorio con un gráfico del error cuadratico VS numero de epoch (../SPAM.png).

# Referencias
## TF-IDF
https://gist.github.com/guenodz/d5add59b31114a3a3c66

## Stemming y Stopwords
https://raw.githubusercontent.com/harryaskham/Twitter-L-LDA/master/util/Stemmer.java

