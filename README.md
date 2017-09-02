# neural-network
Implementación de red neuronal escrita en Java; con documentación en Español.
## Reporte
En el contexto de la Tarea 1 del curso CC5114 del 2017, se presenta el [REPORTE](https://github.com/Vichoko/neural-network/blob/master/REPORT.md).

# Requerimientos
* JUnit 4
* Java 1.7+

# Generalidades
* Las neuronas tienen función de activación sigmoidea; se puede extender a otras funciones.
* La red neuronal soporta vectores de entrada de tamaño fijo; el cual se determina en el constructor.
* La red neuronal provee flexibilidad en la cantidad de capas y neuronas por capa.
* La red neuronal permite obtener vector de salida (de capa de salida) de tamaño fijo (detemrinado por cantidad de neuronas de esta).

# Uso
## Crear red
### Constructor
```Java
	public NeuralNetwork(double learningRate) {...}
	public NeuralNetwork() {...} // learning rate de 0.1
```

### Crear capas
```Java
	public void newInputLayer(int inputSize, int numberOfNeurons) throws Exception {...} // Crear capa de entrada
	public void newHiddenLayer(int numberOfNeurons) throws Exception {...} // crear capa oculta o de salida
	public void closeNetwork() throws Exception {...} // se transforma ultima capa en capa de salida
```

### Entrenar
```Java
	public void train(double[][] input, double[][] expectedOutput, int nEpochs) throws Exception {...} /** input y expectedOutput deben tener la misma cantidad de elementos, 
	nEpochs es la cantidad de veces que se entrenara con el dataSet.*/

```

### Predecir
Metodo para hacer una prediccion individual, sobre un input cualquiera.

```Java
	public double[] predict(double[] input) throws Exception {...} /** retorna vector de predicciones (con valores entre 0 y 1) con tamaño igual a la cantidad de neuronas de la capa de salida */
	public int[] binaryPredict(double[] input, double threshold) throws Exception {...} /** Metodo para forzar predicciones binarias, se evalua cada elemento de la predicción mediante el threshold y se deja un valor 0 o 1 en el vector */
	
```

### Metricas binarias
El paquete provee un metodo para obtener gran cantidad de metricas binarias, a partir de un data-set de pruebas.
Las metricas las retorna en un HashMap indexado por el nombre de la metrica.

#### Metricas disponibles
* tasa de aciertos (verdaderos positivos y verdaderos negativos) [tasa_aciertos]
* tasa de desaciertos (falsos positivos y falsos negativos) [tasa_desaciertos]
* recall [recall]
* precision [precision]

#### Requisitos
* Conjunto de datos de entrada con sus respectivas etiquetas (1 o 0).

#### Uso
```
		double threshold = 0.25;
		boolean verbose = true;
		HashMap<String, Double> metricsData = utils.binaryMetrics(net,
				testData,
				testClasses, 
				threshold,
				verbose); // verbose para imprimir en consola mayor informacion
				
```
### Ejemplo
```Java
		NeuralNetwork net = new NeuralNetwork(0.1); // learningRate = 0.1
		net.newInputLayer(2, 2); // 1° argumento: n_inputs; 2°: cantidad de neuronas en capa
		net.newHiddenLayer(3); // 3 neuronas en capa oculta
		net.newHiddenLayer(1); // 1 neurona en capa oculta
		net.closeNetwork(); // se cierra red y ultima capa queda como capa de salida (1 salida, en este ejemplo).
```
Más ejemplos en ```LayerNetworkTest.java```, ```spam.MainClass.java```


# Tests
Se hace aprender a red neuronal varias compuertas logicas (AND, XOR, OR). Además se hace aprender a clasificar puntos sobre (y bajo) una funcion lineal.
Al ejecutar los test, se exportan gráficos de su error cuadratico VS numero de epoch al directorio del proyecto, para poder visualizar como la red neuronal aprende.

Tests:
* LayerNetworkTest.java
* NeuralTest.java
* utilsTest.java

# Clasificar mensajes de Spam (Clasificacion de texto)
* Para información especifica sobre mi experimento de **Clasificador de SMS de Spam** en [../src/spam/README.md](https://github.com/Vichoko/neural-network/blob/master/src/spam/README.md)

# Clasificar venenocidad de hongos
* Para información especifica sobre mi experimento de **Clasificador de toxicidad de hongos** en [../src/fungi/README.md](https://github.com/Vichoko/neural-network/blob/master/src/fungi/README.md);