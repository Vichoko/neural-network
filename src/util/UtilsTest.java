package util;

import static org.junit.Assert.*;

import org.junit.Test;
public class UtilsTest {
	
	
	
	@Test
	public void dotProductTest() throws Exception {
		assertEquals(utils.dotProduct(new double[] {2,2},new double[] {1,1}), 
				utils.dotProduct(new double[] {1,1},new double[] {2,2}),0.5);
		assertEquals(utils.dotProduct(new double[] {2,2},new double[] {1,1}), 4, 0.5);
				
	}
	
	
	@Test
	public void normalizationTest() {
		double[][] input= new double[][] {{-10, -2}, {-2, 3, 5}, {9}, {1, 2, 4, 10}, {-3, -2}};
		double[][] normalized = utils.normalize(input, -10, 10);
		double[][] denormalized = utils.denormalize(normalized, -10, 10);
		
		for (double[] vector : normalized) {
			for (double number : vector) {
				assertTrue(number >= 0 && number <= 1);
			}	
		}
		
		int inputIndex = 0;
		for (double[] vector : denormalized) {
			int vectorIndex = 0;
			for (double number : vector) {
				assertEquals(number, input[inputIndex][vectorIndex++], 0.01);
			}
			inputIndex++;
		}
		
		
	}
	

}
