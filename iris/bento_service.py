import pandas as pd
import json
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True) #打包时自动寻找所有的依赖
@artifacts([SklearnModelArtifact('StackingClassifier'),SklearnModelArtifact("GaussianNB")])#定义的两个模型
class IrisClassifier(BentoService):
    @api(input=JsonInput(), batch=False)#使用jsoninput，batch注意为false
    def predict(self, parsed_json):
        df = pd.read_json(parsed_json,orient='table')#将json转换为dataframe
        data1 = df[["Aggression", "pressure", "anxiety", "suspect", "balance","confidence", "energy", "self-regulation", "inhibition", "neuroticism","deperession", "happiness", "fh_m", "fh_s","I","E"]]#选取特征列
        result = pd.DataFrame([['StackingClassifier', self.artifacts.StackingClassifier.predict(data1),'HealthType=2'], ['GaussianNB', self.artifacts.GaussianNB.predict_proba(data1)[:, 0],'Probability']])
        return json.loads(result.to_json(orient="records")) #按行导出json
