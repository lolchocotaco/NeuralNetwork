class node(numPrevNodes: Int) {
	var inputW = Array.fill[Double](numPrevNodes)(0)
	
	def setWeight(ind: Int, weight: Double){
	    this.inputW.update(ind, weight)
	}
}