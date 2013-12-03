import scala.io.Source

class nnet(pathName: String) {
	
    var nodesIn = 0
    var nodesOut = 0
    var nodesH = 0
    
    def printVals() = println(this.nodesIn + " " + this.nodesH+ " " + this.nodesOut)
    
   // Load for network from file.
    def init() = {
        println(pathName)
        val source = Source.fromFile(pathName)
		val lines  = source.getLines()
		val numNodes= (lines.next).split(" ")
		this.nodesIn = numNodes(0).toInt
		this.nodesH = numNodes(1).toInt
		this.nodesOut = numNodes(2).toInt
		source.close()
    }
    

}