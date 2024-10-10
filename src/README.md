# Sample Code

## Scenario

1. **Product Technical Documentation**

   - 概要：製品・サービスにおける詳細仕様を説明するテクニカルドキュメントの内容をもとに回答するRAGアプリケーション
   - 目的：構造化されたドキュメントを利用して、基本的なRAGアプリケーションを構築する
   - 利用データ：[Azure AI Search のテクニカルドキュメント](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search)
   - ファイル構造: 
     - `01_Preprocessing.ipynb`: ドキュメントをOCRして文章と画像を抽出する。
     - `01_IndexingPushType.ipynb`: `01_Preprocessing.ipynb` で抽出した文章と画像をインプットにして、Azure AI Search に検索インデックスを構築し、SDKを利用してPUSH型でデータをインポートする。さらにRGAパイプラインを定義して手動評価をする。
     - `01_IndexingPullType.ipynb`: `01_Preprocessing.ipynb` で抽出した文章と画像をインプットにして、Azure AI Search に検索インデックスを構築し、インデクサーを利用してPULL型でデータをインポートする。さらにRGAパイプラインを定義して手動評価をする。

2. **[WIP]Search for Article**

   - 概要：特定ドメイン（ここではLLM関連）の論文の内容をもとに回答するRAGアプリケーション
   - 目的：専門性の高い特定ドメインにおけるRAGアプリケーションを構築する
   - 利用データ：WIP

3. **Business Documents**

   - 概要：ビジネスドキュメント（pptx, pdf, word, etc...）の内容をもとに回答するRAGアプリケーション
   - 目的：ビジネスにおける多様なドキュメント（内容・形式）を利用して、RAGアプリケーションを構築する
   - 利用データ：内閣府の[AI戦略会議](https://www8.cao.go.jp/cstp/ai/index.html)の各種資料
   - ファイル構造: 
     - `03_Preprocessing.ipynb`: ドキュメントをOCRして文章と画像を抽出する。
     - `03_IndexingPushType.ipynb`: `03_Preprocessing.ipynb` で抽出した文章と画像をインプットにして、Azure AI Search に検索インデックスを構築し、SDKを利用してPUSH型でデータをインポートする。
     - `03_Query.ipynb`:　`03_IndexingPushType.ipynb` で構築した検索インデックスに対してRAGパイプラインを定義し、それを Azure AI の Evaluation 機能で評価する。