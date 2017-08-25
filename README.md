# neural-network
Implementación de red neuronal escrita en Java.

# Requerimientos
* JUnit 4
* Java 1.7+

# Generalidades
* Las neuronas tienen función de activación sigmoidea; se puede extender a otras funciones.

# Uso
## Crear red
### Constructor
```Java
	public NeuralNetwork(double learningRate) {...}
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
```Java
	public double[] predict(double[] input) throws Exception {...} /** recibe vector de predicciones (con valores entre 0 y 1) con tamaño igual a la cantidad de neuronas de la capa de salida */
	public int[] binaryPredict(double[] input, double threshold) throws Exception {...} /** Metodo para forzar predicciones binarias, se evalua cada elemento de la predicción mediante el threshold y se deja un valor 0 o 1 en el vector */
	
```
### Ejemplo
```Java
		NeuralNetwork net = new NeuralNetwork(0.1); // learningRate = 0.1
		net.newInputLayer(2, 2); // 1° argumento: n_inputs; 2°: cantidad de neuronas en capa
		net.newHiddenLayer(3); // 3 neuronas en capa oculta
		net.newHiddenLayer(1); // 1 neurona en capa oculta
		net.closeNetwork(); // se cierra red y ultima capa queda como capa de salida (1 salida).
```
Más ejemplos en ```LayerNetworkTest.java```


# Tests
Ejecutar:
* LayerNetworkTest.java
* NeuralTest.java

