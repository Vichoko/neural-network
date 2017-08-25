package network;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.junit.Before;
import org.junit.Test;

public class LayerNetworkTest {

	NeuralNetwork little;
	NeuralNetwork bigger;
	@Before
	public void setUp() throws Exception {
		little = new NeuralNetwork();
		little.newInputLayer(2, 1);
		little.closeNetwork();
		
		bigger = new NeuralNetwork();
		bigger.newInputLayer(3,2);
		bigger.newHiddenLayer(5);
		bigger.newHiddenLayer(10);
		bigger.closeNetwork();
		
	}

	@Test
	public void NetworkTest() throws Exception {
		assertEquals(little.forwardFeed(new double[] {1,2}).length, 1);
		assertEquals(bigger.forwardFeed(new double[] {1,2,4}).length, 10);
		bigger.backPropagation(new double[] {1,2,3,4,5,6,7,8,9,10});
		little.backPropagation(new double[] {1});
	}
	@Test
	public void NetworkXORLearningTest() throws Exception{
		NeuralNetwork net = new NeuralNetwork(0.1);
		net.newInputLayer(2, 2);
		net.newHiddenLayer(3);
		net.newHiddenLayer(1);
		net.closeNetwork();
		
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender XOR.
		 * Clases binarias: 1 si las bit_1 XOR bit_2 == 1; 0 si no.*/
		
		// Pocas combinaciones posibles, mas enfasis al numero de epochs 
		double[][] input = new double[4][2];
		double[][] expectedOutput = new double[4][1];

		
		input[0] = new double[] {0,0};
		input[1] = new double[] {0,1};
		input[2] = new double[] {1,0};
		input[3] = new double[] {1,1};

		expectedOutput[0] = new double[] {0};
		expectedOutput[1] = new double[] {1};
		expectedOutput[2] = new double[] {1};
		expectedOutput[3] = new double[] {0};
		
		net.train(input, expectedOutput, 3000000);
		
		/** Calcular tabla de aciertos
		 * 
		 * verdaero = 1; falso = 0 */
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		int casosTotales = 300000;
		for (int i = 0; i < casosTotales; i++) {
			double seed = Math.random();
			int randomIndex;
			if (seed < 0.25)
				randomIndex = 0;
			else if (seed < 0.5)
				randomIndex = 1;
			else if (seed < 0.75)
				randomIndex = 2;
			else
				randomIndex = 3;
			
			int[] realOutput = net.binaryPredict(input[randomIndex], 0.5);
			// expected class is 0
			if (expectedOutput[randomIndex][0] == 0 && 0 == realOutput[0]) 
				verdaderosNegativos++;
			else if (expectedOutput[randomIndex][0] == 0 && 1 == realOutput[0]) 
				falsosNegativos++;
			else if (expectedOutput[randomIndex][0] == 1 && 1 == realOutput[0])
				verdaderosPositivos++;
			else if (expectedOutput[randomIndex][0] == 1 && 0 == realOutput[0])
				falsosPositivos++;
			else
				System.out.println("expected is "+expectedOutput[i][0]+" real is "+realOutput[0]);
					
		}
		
		System.out.println("Numero de experimentos: " + casosTotales);
		System.out.println("Verdaderos Positivos: " + verdaderosPositivos);
		System.out.println("Verdaderos Negativos: " + verdaderosNegativos);
		System.out.println("Falsos Positivos: " + falsosPositivos);
		System.out.println("Falsos Negativos: " + falsosNegativos);
		double tasaAciertos = (verdaderosPositivos + verdaderosNegativos)*1.0/casosTotales;
		double tasaDesaciertos = (falsosPositivos + falsosNegativos)*1.0/casosTotales;
		System.out.println("Tasa aciertos: " + tasaAciertos + "; tasa desaciertos: " + tasaDesaciertos);
		assertTrue(tasaAciertos > 0.70);
		assertTrue(tasaDesaciertos < 0.20);

		
		
	}
	
	@Test
	public void NetworkLinearFunctionLearningTest() throws Exception {
		NeuralNetwork net = new NeuralNetwork(0.2);
		net.newInputLayer(2, 2);
		net.newHiddenLayer(2);
		net.newHiddenLayer(1);
		net.closeNetwork();
		/** Red con 2 neuronas de entrada, 2 escondidas y una de salida, generada con pesos aleatorios, para aprender una función linear.
		 * Clases binarias: 1 si el punto (x,y) esta sobre la recta, 0 si no.*/
		
		// generar datos de entrenamiento y test
		int dataSetSize = 10000;
		double[][] input = new double[dataSetSize][2];
		double[][] expectedOutput = new double[dataSetSize][1];
		for (int i = 0; i < dataSetSize; i++) {
			/** Fun : Y = -1*X
			 * Si punto esta abajo+izquierda, clase 0.
			 * Si punto esta arriba+derecha, clase 1.*/
			input[i] = new double[] {Math.random()*100-50, Math.random()*100-50};
			expectedOutput[i] = new double[] {NeuralTest.abovenrightFunction(input[i][0], input[i][1]) ? 1 : 0};						
		}
		
		// entrenar con mitad
		
		
		net.train(Arrays.copyOfRange(input, 1, (int)Math.floor(dataSetSize/2)), 
				Arrays.copyOfRange(expectedOutput, 1, (int)Math.floor(dataSetSize/2)), 
				5000);
		
		// evaluar con otra mitad disjunta
		/** Calcular tabla de aciertos
		 * 
		 * verdaero = 1; falso = 0 */
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		int casosTotales = dataSetSize - ((int)Math.floor(dataSetSize/2) + 1);
		for (int i = (int)Math.floor(dataSetSize/2) + 1; i < dataSetSize; i++) {
			int[] realOutput = net.binaryPredict(input[i], 0.5);
			// expected class is 0
			if (expectedOutput[i][0] == 0 && 0 == realOutput[0]) 
				verdaderosNegativos++;
			else if (expectedOutput[i][0] == 0 && 1 == realOutput[0]) 
				falsosNegativos++;
			else if (expectedOutput[i][0] == 1 && 1 == realOutput[0])
				verdaderosPositivos++;
			else if (expectedOutput[i][0] == 1 && 0 == realOutput[0])
				falsosPositivos++;
			else
				System.out.println("expected is "+expectedOutput[i][0]+" real is "+realOutput[0]);
		}
		
		System.out.println("Numero de experimentos: " + casosTotales);
		System.out.println("Verdaderos Positivos: " + verdaderosPositivos);
		System.out.println("Verdaderos Negativos: " + verdaderosNegativos);
		System.out.println("Falsos Positivos: " + falsosPositivos);
		System.out.println("Falsos Negativos: " + falsosNegativos);
		double tasaAciertos = (verdaderosPositivos + verdaderosNegativos)*1.0/casosTotales;
		double tasaDesaciertos = (falsosPositivos + falsosNegativos)*1.0/casosTotales;
		System.out.println("Tasa aciertos: " + tasaAciertos + "; tasa desaciertos: " + tasaDesaciertos);
		assertTrue(tasaAciertos > 0.70);
		assertTrue(tasaDesaciertos < 0.20);
	
		
				
		
		
	}
	@Test
	public void LayerTest() throws Exception {
		NeuralLayer first = new NeuralLayer(1,1);
		NeuralLayer second = new NeuralLayer(2,2);
		assertEquals(first.getOutputSize(), 1);
		assertEquals(second.getOutputSize(), 2);
		double[] fres = first.synapsis(new double[] {1});
		double[] sres = second.synapsis(new double[] {2, 3});
		assertEquals(first.getPastOutputs().length, first.getOutputSize());
		assertEquals(second.getPastOutputs().length, second.getOutputSize());
		assertEquals(first.getPastOutputs(), fres);
		assertEquals(second.getPastOutputs(), sres);
		}

}
