import scala.io.Source
import java.nio.file._

object main {
    
    def getFile(): String  = {
    	var fileName = readLine()
        while(!Files.exists(Paths.get(fileName))){
            println("Please input a valid filename")
            fileName = readLine()
        }
        return fileName
    }
    
    
    def main(args: Array[String]) {
	    //Gets arguments from user
        var netLoc, trainLoc,outName = ""
        var epoch: Int = 0
        var rate: Double = 0.0
        
	    println("Initial Net: ")
//		var netLoc = getFile
		println("Training Set: ")
//		var trainLoc = getFile
		println("Output Name: ")
//		var outName = getFile
		println("Epochs: ")
//		var epoch : Int = readInt()
		println("Learning Rate: ")
//		var rate : Double = readDouble()
		  
		
		// Just for testing purposes
		netLoc = "resource/sample.NNWDBC.init"
		trainLoc = "resource/wdbc.train"
		outName = "resource/test.out"
		epoch = 100
		rate = 2
		//
		
		var net = new nnet(netLoc)
        net.init
        net.printVals
        
        var node = new node(3)
        println(node.inputW(0))
        node.setWeight(0, 10.0)
        println(node.inputW(0))
			   
    }

}