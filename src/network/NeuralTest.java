package network;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;

public class NeuralTest {
	
	Neuron and(){
		return new Neuron(-3, new double[] {2, 2});
	}
	
	Neuron nand(){
		return new Neuron(3, new double[] {-2, -2});
	}
	
	Neuron or(){
		return new Neuron(0, new double[] {1, 1});
	}
	
	Neuron pAND;
	Neuron pNAND;
	Neuron pOR;

	public int[] sum(int a, int b) throws Exception{
		int outa = pNAND.binarySynapsis(new double[] {a, b});
		int outb = pNAND.binarySynapsis(new double[] {a, outa});
		int outc = pNAND.binarySynapsis(new double[] {outa, b});
		int outd = pNAND.binarySynapsis(new double[] {outb, outc});
		int oute = pNAND.binarySynapsis(new double[] {outa, outa});
		return new int[] {outd, oute};
		
	}
	@Before
	public void setUp() throws Exception {
		pAND = and();
		pOR = or();
		pNAND = nand();
	}

	@Test
	public void testAND() throws Exception {
		// tabla de verdad
		assertEquals(pAND.binarySynapsis(new double[] {0,0}), 0);
		assertEquals(pAND.binarySynapsis(new double[] {0,1}), 0);
		assertEquals(pAND.binarySynapsis(new double[] {1,0}), 0);
		assertEquals(pAND.binarySynapsis(new double[] {1,1}), 1);
	}	


	@Test
	public void testNAND() throws Exception {
		assertEquals(pNAND.binarySynapsis(new double[] {0,0}), 1);
		assertEquals(pNAND.binarySynapsis(new double[] {0,1}), 1);
		assertEquals(pNAND.binarySynapsis(new double[] {1,0}), 1);
		assertEquals(pNAND.binarySynapsis(new double[] {1,1}), 0);
	}	
	@Test
	public void testOR() throws Exception {
		assertEquals(pOR.binarySynapsis(new double[] {0,0}), 0);
		assertEquals(pOR.binarySynapsis(new double[] {0,1}), 1);
		assertEquals(pOR.binarySynapsis(new double[] {1,0}), 1);
		assertEquals(pOR.binarySynapsis(new double[] {1,1}), 1);
	}
	@Test
	public void testSum() throws Exception {
		// primero es sum, segundo es carry
		assertEquals(sum(0,0)[0], 0);
		assertEquals(sum(0,0)[1], 0);
		
		assertEquals(sum(1,0)[0], 1);
		assertEquals(sum(1,0)[1], 0);
		
		assertEquals(sum(0,1)[0], 1);
		assertEquals(sum(0,1)[1], 0);
		
		assertEquals(sum(1,1)[0], 0);
		assertEquals(sum(1,1)[1], 1);
	}
	
	static boolean abovenrightFunction(double x, double y) {
		// Funcion es recta diagonal que corta plano en mitades, desde izq-arriba a der-abajo
		// Y = -1*X
		double image = -1*x;
		if (image - y < 0) 
			return true;
		else
			return false;
		
	}
	
	@Test
	public void testTraining() throws Exception{
		// Puntos x,y generados entre -50 y 50.
		
		
		int initialW = 1;
		int initialBias = 2;
		System.out.println("Starting training with initial weight = " + initialW + " and bias = " + initialBias);
		Neuron p = new Neuron(initialBias,new double[] {initialW,initialW});
		int desiredOutput;
		double learnRate = 0.1;
		int iterations = 10000000;
		for (int i = 0; i < iterations; i++) {
			/** Fun : Y = -1*X
			 * Si punto esta abajo+izquierda, clase 0.
			 * Si punto esta arriba+derecha, clase 1.*/
			double anyX = Math.random()*100-50;
			double anyY = Math.random()*100-50;
			
			if (!abovenrightFunction(anyX, anyY)) {
				// expected class is 0
				desiredOutput = 0;				
			} else {
				// expected class is 1
				desiredOutput = 1;
			}
			p.singleTrain(learnRate, new double[] {anyX, anyY}, desiredOutput);
		}
		System.out.println("Finished training with wx = " + p.getWeights()[0] + "; wy = " + p.getWeights()[1] + "; bias = " + p.getBias());
		
		/** Calcular tabla de aciertos
		 * 
		 * verdaero = 1; falso = 0 */
		int verdaderosPositivos = 0;
		int verdaderosNegativos = 0;
		int falsosPositivos = 0;
		int falsosNegativos = 0;
		int casosTotales = 100000;
		for (int i = 0; i < casosTotales; i++) {
			double anyX = Math.random()*100-50;
			double anyY = Math.random()*100-50;
			int prediction = p.binarySynapsis(new double[] {anyX, anyY});
			if (!abovenrightFunction(anyX, anyY)) {
				// expected class is 0
				desiredOutput = 0;
				if (desiredOutput == prediction) 
					verdaderosNegativos++;
				 else 
					falsosNegativos++;
			} else {
				// expected class is 1
				desiredOutput = 1;
				if (desiredOutput == prediction)
					verdaderosPositivos++;
				else
					falsosPositivos++;
			}			
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
	

	}
	


