# Best Practices for RAG (Retrieval-Augmented Generation)

本ドキュメントでは、多様なデータタイプ（テキスト、画像、テーブル）が混在するドキュメントにおいて、RAG を実現するための設計ガイダンスを提供します。

## Retriever

### Option 1: Multi-modal embedding retrieval

マルチモーダルの Embedding モデル (Azure AI Vision, CLI など) を使用して、画像とテキストを一緒に埋め込みます。クエリを同一のEmbedding モデルで埋め込んで、ベクトル検索を使用して画像もしくはテキストチャンクを取得します。取得した画像とテキストチャンクをマルチモーダル LLM （GPT-4V, GPT-4oなど）に渡して回答を生成します。

- Pro: Potential for highest quality retrieval b/c images are directly embedded
- Con: Fewer options for multi-modal embeddings

### Option 2: Produce image summaries

マルチモーダル LLM(GPT-4V, GPT-4oなど)を使用して、画像からテキストでキャプション・サマリを生成します。テキスト用の Embedding モデルを使用して、キャプション・サマリを埋め込んだり取得したりします。そして、取得したテキストをLLMに渡して回答を生成します。

- Pro: Simplicity, as it uses text embeddings and does not rely on a multimodal LLM for answer synthesis
- Con: Information loss as images are not used directly in answer synthesis or retrieval

### Option 3: Retrieve w/ image summaries, but pass images for synthesis

マルチモーダル LLM(GPT-4V, GPT-4oなど)を使用して、画像からテキストでキャプション・サマリを生成します。テキスト用の Embedding モデルを使用して、キャプション・サマリを埋め込んだり取得したりします。また、検索インデックス内で、キャプション・サマリのデータと生の画像への参照リンクを関連付け、生の画像データを取得できるようにします。テキストと生の画像データをマルチモーダル LLM （GPT-4V, GPT-4oなど）に渡して回答を生成します。

- Pro: Text embeddings simplify retrieval but it still uses image in answer synthesis
- Con: Potential quality loss in retrieval b/c images summaries are embedded