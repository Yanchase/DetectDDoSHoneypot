import pandas as pd
import json
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('svmclf')])
class SVMClassifier(BentoService):
    @api(input=DataframeInput(), batch=True)
    def predict(self, parsed_json):
        df = pd.read_json(parsed_json, orient='table')  # 将json转换为dataframe
        data1 = df[
            ["Fwd Packet Length Max", "wd Packet Length Mean", "Avg Fwd Segment Size", "act_data_pkt_fwd", "Total Length of Fwd Packets", "Subflow Fwd Bytes", "Init_Win_bytes_forward"]]
        # 选取特征列
        result = pd.DataFrame(['SVMclf', self.artifacts.svmclf.predict_proba(data1)[:, 0], 'Label=1'])
        return json.loads(result.to_json(orient="records"))  # 按行导出json
