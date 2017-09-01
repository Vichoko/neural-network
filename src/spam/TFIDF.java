package spam;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
/**
* @author Mohamed Guendouz
*/
public class TFIDF {
	HashMap<String, Double> idfCache;

	    /**
	     * @param doc  list of strings
	     * @param term String represents a term
	     * @return term frequency of term in document
	     */
	    public double tf(List<String> doc, String term) {
	        double result = 0;
	        for (String word : doc) {
	            if (term.equalsIgnoreCase(word))
	                result++;
	        }
	        return result / doc.size();
	    }

	    /**
	     * @param docs list of list of strings represents the dataset
	     * @param term String represents a term
	     * @return the inverse term frequency of term in documents
	     */
	    public double idf(List<List<String>> docs, String term) {
	    	if (idfCache.containsKey(term)) {
	    		return idfCache.get(term);
	    	}
	    	
	        double n = 0;
	        for (List<String> doc : docs) {
	            for (String word : doc) {
	                if (term.equalsIgnoreCase(word)) {
	                    n++;
	                    break;
	                }
	            }
	        }
	        double idf = Math.log(docs.size() / n);
	        idfCache.put(term, idf);
	        return idf;
	    }

	    /**
	     * @param doc  a text document
	     * @param docs all documents
	     * @param term term
	     * @return the TF-IDF of term
	     */
	    public double tfIdf(List<String> doc, List<List<String>> docs, String term) {
	    	idfCache = new HashMap<>();
	        return tf(doc, term) * idf(docs, term);

	    }


	    


	}


