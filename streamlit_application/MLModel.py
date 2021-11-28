import pandas as pd
from transformers import pipeline

class MLModel:
    def __init__(self):
        self.roberta_findings=[]

    def RobertaModel(self,TopNDf,query):
        '''
        This Function runs the RobertaModel pretrained on the SQUAD2.0 dataset.
        return: all the answers along with score with respect to query
        '''
        qa_pipeline = pipeline("question-answering",model="deepset/roberta-base-squad2",tokenizer="deepset/roberta-base-squad2", revision="v1.0")
        #looping through all context
        for x,context in enumerate(TopNDf['Context']):
          prediction = qa_pipeline({'context': context,'question': query})
          self.roberta_findings.append([prediction['answer'],prediction['score'],TopNDf.iloc[x,2],context])
        return self.roberta_findings

    def TopNFindings(self,top_n=3):
        '''
        This Function finds the topN predictions based on the query
        return: TopN findings
        '''
        return sorted(self.roberta_findings, key=itemgetter(1), reverse=True)[:top_n]

    def ConverttoDf(self):
        '''
        This Function converts the findings list to dataframe
        return: dataframe
        '''
        Converted_Df = pd.DataFrame(self.roberta_findings,columns=['Prediction','Score','Wiki_Page','Context'])
        return Converted_Df

    def TopNDf(self,Df,top_n=3):
        '''
        This Function finds the topN predictions on th dataframe
        return: datafrmae with TopN findings
        '''
        return Df.sort_values(by=['Score'], ascending=False).iloc[:top_n,]