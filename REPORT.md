# Reporte Tarea 1
Vicente Oyanedel Muñoz
18.933.042-1
https://github.com/Vichoko/neural-network

## Generalidades
Este reporte incluye todo lo requerido. Para más información, puede encontrarla en los siguientes links:
* Para información sobre **como usar la red neuronal, ejecutar sus pruebas, ejemplos** y más [../README.md](https://github.com/Vichoko/neural-network/blob/master/README.md).
* Para información especifica sobre mi experimento de **Clasificador de SMS de Spam** en [../src/spam/README.md](https://github.com/Vichoko/neural-network/blob/master/src/spam/README.md)
* Para información especifica sobre mi experimento de **Clasificador de toxicidad de hongos** en [../src/fungi/README.md](https://github.com/Vichoko/neural-network/blob/master/src/fungi/README.md);

## Introducción
Este proyecto consiste de la implementación de una red neuronal y pruebas de su funcionamiento.

La primera gran parte del proyecto consiste en la **implementación de la red, sus capas y las neuronas**. Aun más importante es la batería de tests adjunta con el proyecto; en la cual entrena la red neuronal con diversos problemas conocidos y luego se mide su desempeño. Esta batería de tests debería poder ejecutarse en su copia del repositorio.

La segunda parte corresponde a **experimentos hechos con data-set externos**. En los cuales gran parte de la implementación corresponde a:
1. Pre-procesamiento de los datos, para transformarlos al formato de entrada de la red. 
2. La configuración de la red neuronal, para optimizar el aprendizaje mediante prueba y error.
3. La captura de métricas de desempeño, múltiples reiteraciones.
4. Conclusiones acerca del problema, los datos y las métricas recibidas.

## Pruebas
Las pruebas que no respectan a la red neuronal directamente son para comprobar el correcto funcionamiento de funciones auxiliares y partes más pequeñas del sistema.

Las pruebas de la red neuronal consiste en problemas simples de clasificación binaria. Donde se predice con un solo output (1 o 0) y, además, pruebas que se predice con 2 output (Se interpreta según el índice del valor más alto). Usted debería poder ejecutar las pruebas y comprobar la alta tasa de precisión que alcanza la red neuronal.
**Al ejecutar las pruebas se guarda un gráfico con el error medio VS numero de epoch para cada clasificicación, en el directorio del proyecto**.
### Clasificador AND 
![AND](https://i.imgur.com/kezrOGg.png)
### ClasificadOR
![OR](https://i.imgur.com/k5fTuwl.png)
### Clasificador XOR
![XOR](https://i.imgur.com/QWKzQAQ.png)
### Clasificador arriba/abajo de función lineal
![Arriba/Abajo de función lineal](https://i.imgur.com/93XYVjF.png)


## Mis experimentos 
### Clasificador de SMS de Spam

#### Motivación
Dado un SMS (Texto plano), lograr clasificarlo como spam (o no spam).

#### Datos
Los datos (https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip) corresponden a un archivo TSV (Tab Separated Values).
##### Ejemplo
```
...
ham		I HAVE A DATE ON SUNDAY WITH WILL!!
spam	XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL
ham		k...i'm watching here:)
...
```

#### Pre-procesamiento de Texto
1. Transformar clases "ham" y "spam" a valores 0 y 1; respectivamente.
2. Transformar SMS (texto) a una 'bag of words' (Matriz de SizeOf(DataSet) x SizeOf(Diccionario)) de mínimo tamaño posible.

Esta parte del programa basta que se ejecute una vez, dado que el resultado del procesamiento queda guardado en 3 archivos temporales en el directorio del proyecto (../temp_tdidf_*). Si hay algún problema y se quiere forzar el recalculo de este proceso basta con eliminar estos 3 archivos temporales.

Para acercarse al punto 2 se hicieron varios procesos:

##### Quitar Stop-words
Se eliminaron de los SMS palabras que no tienen relevancia en el ingles. Por ejemplo, "a", "the", etc. 
Esto con el fin de disminuir el tamaño del diccionario.

##### Proceso de "Stemming"
Reducir las palabras similares a su raíz; por ejemplo, "run", "running", "runned" a solamente "run".
Esto disminuye el tamaño del diccionario.

##### Quitar caracteres especiales y llevar todo a minúsculas
Esto con el fin de disminuir el tamaño del diccionario. Suponiendo que las letras tienen mayor semántica, en una palabra, que los números y símbolos especiales. Y que las palabras en minúsculas y mayúsculas significan lo mismo.

##### Proceso de calculo de frecuencias normalizadas con TF-IDF
El concepto de "Bag of Words" corresponde a almacenar, para cada texto (fila), la frecuencia de cada uno de sus términos (columna).
Se aplica TF-IDF sobre cada tupla (SMS, termino, data-set), para calcular frecuencia de termino * frecuencia inversa de documentos, con el fin de normalizar los valores. Obteniendo valores más grandes para palabras con poca frecuencia global en los SMS, pero con valores más pequeños para palabras comunes.

#### Procesamiento de datos
Al analizar el resultado del pre-procesamiento anterior, se notó un fuerte des balance entre los SMS etiquetados como SPAM (700) y los que no (5000).
Por lo que se incurrieron en 2 técnicas de "Sampling" para transformar el dataSet de entrenamiento y prueba.

##### Over-Sampling
Balancear la cantidad de datos por cada clase, a la cantidad de la clase mayoritaria.
Para la clase minoritaria, se repiten pares (data, class) elegidos aleatoriamente hasta que ambas clases tienen la misma cantidad de datos.

##### Sub-Sampling
Balancear la cantidad de datos por cada clase, a la cantidad de la clase minoritaria.
Para la clase mayoritaria, se omiten datos para que ambas clases estén igualmente distribuidas en el data-set.

#### Entrenamiento de red neuronal
Se probaron diversas configuraciones, la que se obtuvo mejor resultados consistió en:
* Capa de entrada de Diccionario.size() inputs y neuronas.
* Capa oculta de 20 neuronas.
* Capa de salida de 1 neurona.

El error cuadrático mínimo que se consiguió fue de 240.
![error cuadrático vs epochs](https://i.imgur.com/0stAdDy.png)*
*Cuando vale 0, es porque dejó de aprender dado que se estabilizó el error.


#### Observaciones y resultado

* Se notó que a partir de la 5 epoch aproximadamente, el error cuadrático dejaba de variar; quedándose en un valor cercano a 240

* El clasificador tiende a clasificar todo como no-spam (con threshold de 0.5)

* El clasificar tiende a clasificar todo como spam (con threshold de 0.25)

* El desempeño no varia sustancialmente si se normaliza el "bag of Words" previamente

* Al variar la cantidad de neuronas por capa no varía el desempeño notablemente


#### Conclusión

Con estos datos no se puede resolver el problema de clasificación de una manera precisa. Lo cual se puede deber a la baja cantidad de datos que se tienen (6000 aproximadamente). Dado que al probar el desempeño de la red neuronal con los test, se puede comprobar que funciona correctamente.

Además puede ser por el fuerte desbalance de clases.


### Ejecución

Para ejecutar el experimento se debe ubicar el archivo de datos (../SMSSpamCollection) en el directorio del proyecto

Luego ejecutar la clase spam.MainClass

Se debería visualizar en consola el progreso del programa (pre-procesamiento, entrenamiento y métricas)

Además quedará un archivo en el directorio con un gráfico del error cuadrático VS numero de epoch (../SPAM.png)


### Referencia

#### TF-ID

https://gist.github.com/guenodz/d5add59b31114a3a3c6


#### Stemming y Stopword

https://raw.githubusercontent.com/harryaskham/Twitter-L-LDA/master/util/Stemmer.jav



### Clasificador de toxicidad de hongos

#### Motivación
Dado caracterizaras de un hongo, poder predecir si es toxico o no.

Además, la red neuronal tiene 2 neuronas de output, por lo que la predicción da 2 números entre 0 y 1. La condición de toxicidad se cumple cuando el segundo numero es mayor que el primero; i.e. output[1] > output[0].


#### Datos
Los datos (https://archive.ics.uci.edu/ml/datasets/Mushroom) corresponden a un archivo CSV (Comma separated value). Donde la primera columna corresponde a si es toxico o no. Y el resto de las columnas corresponden a atributos categóricos, los cuales representan:
```
	1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s 
	2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s 
	3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y 
	4. bruises?: bruises=t,no=f 
	5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s 
	6. gill-attachment: attached=a,descending=d,free=f,notched=n 
	7. gill-spacing: close=c,crowded=w,distant=d 
	8. gill-size: broad=b,narrow=n 
	9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y 
	10. stalk-shape: enlarging=e,tapering=t 
	11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=? 
	12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s 
	13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s 
	14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
	15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y 
	16. veil-type: partial=p,universal=u 
	17. veil-color: brown=n,orange=o,white=w,yellow=y 
	18. ring-number: none=n,one=o,two=t 
	19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z 
	20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y 
	21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y 
	22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
```

##### Ejemplo
```
p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g
e,b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m
p,x,y,w,t,p,f,c,n,n,e,e,s,s,w,w,p,w,o,p,k,s,u
```

#### Pre-procesamiento de datos
Es necesario transformar los atributos categóricos a numéricos para poder ingresarlos a la red neuronal,;para lo cual se decidió utilizar la siguiente representación numérica de los atributos:

Para cada atributo categórico posible, se le asigna un índice único en un arreglo de números binarios, para el cual si es 1, entonces ese atributo está presente en el hongo (0 si no).

Con esto se pasa de tener 22 atributos, con valores posibles variados. A un arreglo de 122 números binarios, representando los valores posibles que puede tomar cada atributo.

Para hacer esta transformación se programó el método ```parseCategoricalToNumerical```.

#### Entrenamiento de red neuronal
Se probaron diversas configuraciones, la que se obtuvo mejor resultados consistió en:
* Capa de entrada de 122 inputs y neuronas (numero de atributos numéricos).
* Capa oculta de 20 neuronas.
* Capa de salida de 2 neurona.

El error cuadrático mínimo que se consiguió fue de 2000.
![error cuadrático vs epochs](https://i.imgur.com/T2T9LF4.png)


#### Observaciones y resultados
* Se notó que a partir de la 5 epoch aproximadamente, el error cuadrático dejaba de variar.
* El clasificador tiende a clasificar todo como no-toxico.
* Al variar la cantidad de neuronas por capa no varía el desempeño notablemente.
* Al variar el 'learn rate' no varía el desempeño.
* Al variar la cantidad de epoch, no mejora el desempeño.
* Al evaluar con los mismos datos de entrenamiento, tiene 50% de precisión (malo).


#### Conclusión
La red neurona clasifica con una efectividad del 50%, lo cual es como un clasificador que dice que no siempre. Esto es muy malo porque alguien podría comer un hongo toxico por culpa de este mal clasificador.

Después de debugear muchas horas, se llegó a la conclusión que el problema radica en como se numerizaron los atributos categóricos. 
Como trabajo futuro queda evaluar con otros métodos para transformar atributos categóricos a numéricos.

### Ejecución
Para ejecutar el experimento se debe ubicar el archivo de datos (../parseCategoricalToNumerical) en el directorio del proyecto.

Luego ejecutar la clase fungi.MainClass.

Se debería visualizar en consola el progreso del programa (entrenamiento y métricas).

Además quedará un archivo en el directorio con un gráfico del error cuadrático VS numero de epoch (../FUNGI.png).

# Desarrollo
Ahora a responder las dudas planteadas para esta tarea:
## How does the number of hidden layers impact the learning rate?
Mientras más numero de capas, más lento se hace el aprendizaje.

## What is the speed of your network to process data?
Depende de la topología elegida. En particular cuando se elije una capa con gran numero de neuronas la velocidad de procesamiento disminuye fuertemente.
Más específicamente, mientras más capas o más neuronas por capas, más lenta la velocidad de procesamiento.

## Effect of different learning rates
Valores muy bajos hace que aprenda excesivamente lento por cada epoch. Aunque con valores muy grandes también dificulta el aprendizaje.
Un valor correcto no es ni muy alto ni muy bajo. En la practica me sirvieron valores entre 0.1 y 0.3.

Esto tiene relación con el método de descenso estocástico del gradiente, donde el 'learning rate' corresponde al tamaño del paso con que desciende en el gradiente. Para pasos muy grandes puede que nunca se alcance un mínimo; en cambio, el método quedará oscilando en torno a este eternamente; incluso podría aumentar el error cuadrático.

## Does the order of the training data matter? 
Si. En general se recomienda hacer un shuffle del dataset antes de entrenar.
Más si se sabe que los datos tienen algún orden en el data-set; por ejemplo, si están ordenados por clase. Esto puede inducir un sesgo en el aprendizaje, y afectar negativamente en las predicciones.

## What are the neurons that changes the most during the learning phase?
Me imagino que las de la capa de input. Dado que, según he leído, ellas son las que separan el problema; cuando el problema es linear-separable.
Las capas intermedias aprenden las curvaturas del problema (así logrando aprender el XOR, por ejemplo). Pero es una conjetura.
