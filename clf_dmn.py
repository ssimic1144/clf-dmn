from sklearn import tree
from collections import defaultdict
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import random
import string
import graphviz

class xmlDmn:
    
    def __init__(self, dmnName, outName = None):
        self.outName = outName
        self.tree = et.parse(dmnName)
        self.xmlRoot = self.tree.getroot() 
        self.namespace = self.xmlRoot.tag.split("}")[0].strip("{")
        self.ns = {"xmlns" : self.namespace}
        self.decisionTableElement = self.xmlRoot.find(".//xmlns:decisionTable", namespaces=self.ns) 

    def printDecisionTable(self):
        """Print decision table - for debugging"""
        for element in self.decisionTableElement:
            if self.namespace in element.tag:
                print(element.tag.split("}")[-1], ":", element.attrib)
    
    def idGen(self, text):
        """Simple ID generator"""
        return text + ''.join(random.choices(string.ascii_lowercase + string.digits, k=7))

    def clearDecisionTable(self):
        """Clears dmn table and prepares it for new input"""
        for element in self.xmlRoot.findall(".//xmlns:decisionTable/", namespaces=self.ns):
            if element.tag == "{%s}input"%(self.namespace):
                self.decisionTableElement.remove(element)
            elif element.tag == "{%s}output"%(self.namespace):
                self.decisionTableElement.remove(element)
            elif element.tag == "{%s}rule"%(self.namespace):
                self.decisionTableElement.remove(element)
    
    def generateTableColumns(self,names):
        """Creates table header """
        outputName = names[-1]
        inputNames = names[:-1]
        for name in inputNames:
            newInput = et.SubElement(self.decisionTableElement, "{%s}input"%(self.namespace), attrib= {"id": self.idGen("input_")})
            newInputExpression = et.SubElement(newInput, "{%s}inputExpression"%(self.namespace), attrib={"id": self.idGen("inputExpression_"),"typeRef":"double"}) 
            newText = et.SubElement(newInputExpression, "{%s}text"%(self.namespace))
            newText.text = name
        et.SubElement(self.decisionTableElement, "{%s}output"%(self.namespace), attrib={"id": self.idGen("output_"),"name": outputName,"typeRef":"string"})
        
    def generateTableRows(self, mlDict):
        for k,v in mlDict:
            newRule = et.SubElement(self.decisionTableElement, "{%s}rule"%(self.namespace), attrib={"id":self.idGen("DecisionRule_")})
            print(k)
            for keyValues in v:
                print(keyValues)
                print(v[keyValues])
                tSignList = list(v[keyValues].keys())
                if len(v[keyValues]) == 0:
                    self.createRuleCell(newRule)               
                if len(v[keyValues]) == 1:    
                    self.createRuleCell(newRule).text = "{} {:.4f}".format(tSignList[0],v[keyValues][tSignList[0]])
                if len(v[keyValues]) == 2:
                    if tSignList[0] == "<=":
                        if v[keyValues][tSignList[0]] > v[keyValues][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.4f}..{:.4f}]".format(v[keyValues][tSignList[1]],v[keyValues][tSignList[0]])
                        else:
                            self.createRuleCell(newRule)
                    if tSignList[0] == ">":
                        if v[keyValues][tSignList[0]] < v[keyValues][tSignList[1]]:
                            self.createRuleCell(newRule).text = "]{:.4f}..{:.4f}]".format(v[keyValues][tSignList[0]],v[keyValues][tSignList[1]])
                        else:
                            self.createRuleCell(newRule)
            newOutEntry = et.SubElement(newRule, "{%s}outputEntry"%(self.namespace), attrib={"id":self.idGen("LiteralExpression_")})
            newOutText = et.SubElement(newOutEntry, "{%s}text"%(self.namespace))
            newOutText.text = str(k)

    def createRuleCell(self,newRule):
        """Creates empty cell"""   
        newInEntry = et.SubElement(newRule, "{%s}inputEntry"%(self.namespace), attrib={"id":self.idGen("UnaryTests_")})
        newInText = et.SubElement(newInEntry,"{%s}text"%(self.namespace))
        return newInText

    def writeTree(self):
        """Save to file"""
        if self.outName is None:
            self.tree.write('new.dmn')
        else:
            self.tree.write(self.outName)

class clfDmn(xmlDmn):

    def __init__(self,csvName, dmnName, target = -1, depth = None, outName = None):
        super().__init__(dmnName, outName)   
        df = pd.read_csv(csvName)   
        self.dfColumns = df.columns
            
        self.dfTarget = df[df.columns[target]]
        self.dfData = df[df.columns.drop(df.columns[target])]
        self.dfFeature = self.dfColumns.drop(self.dfColumns[target])
        self.dfClassColumn = self.dfColumns[target]
        self.dfClassNames = self.dfTarget.unique()
        
        if depth is None:
            self.clf = tree.DecisionTreeClassifier().fit(self.dfData,self.dfTarget)
        else:
            self.clf = tree.DecisionTreeClassifier(max_depth=depth).fit(self.dfData,self.dfTarget)
        
    def visualizeTree(self, name):
        """Creates visualization for tree"""
        dot_data = tree.export_graphviz(self.clf, out_file=None, feature_names=self.dfFeature.astype(str), class_names=self.dfClassNames.astype(str), filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(name)
    
    def generateTableFromClf(self):
        """Extract elements from decision tree and generate dmn table"""
        featureNames = [self.dfFeature[i] for i in self.clf.tree_.feature]  
        leafIds = self.clf.apply(self.dfData) 
        leftChildren = self.clf.tree_.children_left     
        rightChildren = self.clf.tree_.children_right   
        decPath = self.clf.decision_path(self.dfData)   
        threshold = self.clf.tree_.threshold    

        featuresForDiagram = list(self.dfFeature)    
        featuresForDiagram.append(self.dfClassColumn)  

        self.clearDecisionTable()
        self.generateTableColumns(featuresForDiagram)   

        for i in set(leafIds): 
            samplesInNode = decPath.getcol(i).copy()  
            rows = samplesInNode.nonzero()[0]
            sampleId = rows[0]  
            nodeIndex = decPath.indices[decPath.indptr[sampleId]:decPath.indptr[sampleId+1]]   
            className = self.clf.classes_[np.argmax(self.clf.tree_.value[i])]
            inputOutput = defaultdict(dict) 
            for value in featuresForDiagram[:-1]:   
                    inputOutput[className][value] = {}  
            for index, nodeId in enumerate(nodeIndex): 
                nodeFeature = featureNames[nodeIndex[index-1]] 
                nodeThreshold = threshold[nodeIndex[index-1]]  
                if nodeId in set(leftChildren):
                    inputOutput[className][nodeFeature]["<="] = nodeThreshold
                if nodeId in set(rightChildren):
                    inputOutput[className][nodeFeature][">"] = nodeThreshold
            self.generateTableRows(inputOutput.items())     
        self.writeTree()



if __name__ == "__main__":
    newObject = clfDmn(csvName="iris.csv", dmnName="test.dmn")
    #newObject = clfDmn(csvName="winequality-red.csv", dmnName="test.dmn")
    #newObject.visualizeTree("VisualizedTree")
    newObject.generateTableFromClf()
