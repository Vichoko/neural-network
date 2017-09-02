# Clasificador de toxicidad de hongos

## Motivacion
Dado caracterisitcas de un hongo, poder predecir si es toxico o no.

Además, la red neuronal tiene 2 neuronas de output, por lo que la predicción da 2 numeros entre 0 y 1. La condición de toxicidad se cumple cuando el segundo numero es mayor que el primero; i.e. output[1] > output[0].


## Datos
Los datos (https://archive.ics.uci.edu/ml/datasets/Mushroom) corresponden a un archivo CSV (Comma separated value). Donde la primera columna corresponde a si es toxico o no. Y el resto de las columnas coresponden a atributos categoricos, los cuales representan:
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

### Ejemplo
```
p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u
e,x,s,y,t,a,f,c,b,k,e,c,s,s,w,w,p,w,o,p,n,n,g
e,b,s,w,t,l,f,c,b,n,e,c,s,s,w,w,p,w,o,p,n,n,m
p,x,y,w,t,p,f,c,n,n,e,e,s,s,w,w,p,w,o,p,k,s,u
```

## Pre-procesamiento de datos
Es necesario transformar los atributos categoricos a numericos para poder ingresarlos a la red neuronal,;para lo cual se decidió utilizar la siguiente representación numerica de los atributos:

Para cada atributo categorico posible, se le asigna un indice unico en un arreglo de numeros binarios, para el cual si es 1, entonces ese atributo está presente en el hongo (0 si no).

Con esto se pasa de tener 22 atributos, con valores posibles variados. A un arreglo de 122 numeros binarios, representando los valores posibles que puede tomar cada atributo.

Para hacer esta transformación se programó el metodo ```parseCategoricalToNumerical```.

## Entrenamiento de red neuronal
Se probaron diversas configuraciones, la que se obtuvo mejor resultados consistió en:
* Capa de entrada de 122 inputs y neuronas (numero de atributos numericos).
* Capa oculta de 20 neuronas.
* Capa de salida de 2 neurona.

El error cuadratico minimo que se consiguió fue de 2000.
![error cuadratico vs epochs](https://i.imgur.com/T2T9LF4.png)


## Observaciones y resultados
* Se notó que a partir de la 5 epoch aproximadamente, el error cuadratico dejaba de variar.
* El clasificador tiende a clasificar todo como no-toxico.
* Al variar la cantidad de neuronas por capa no varía el desempeño notablemente.
* Al variar el 'learn rate' no varía el desempeño.
* Al variar la cantidad de epoch, no mejora el desempeño.
* Al evaluar con los mismos datos de entrenamiento, tiene 50% de precisión (malo).


## Conclusión
La red neuronal clasifica con una efectividad del 50%, lo cual es como un clasificador que dice que no siempre. Esto es muy malo porque alguien podría comer un hongo toxico por culpa de este mal clasificador.

Después de debugear muchas horas, se llegó a la conclusión que el problema radica en como se numerizaron los atributos categoricos. 
Como trabajo futuro queda evaluar con otros metodos para transformar atributos categoricos a numericos.

# Ejecución
Para ejecutar el experimento se debe ubicar el archivo de datos (../parseCategoricalToNumerical) en el directorio del proyecto.

Luego ejecutar la clase fungi.MainClass.

Se debería visualizar en consola el progreso del programa (entrenamiento y metricas).

Además quedará un archivo en el directorio con un gráfico del error cuadratico VS numero de epoch (../FUNGI.png).


