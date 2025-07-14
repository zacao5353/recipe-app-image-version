from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
import numpy as np
from PIL import Image
import io

# --- ヘルパー関数 ---
def clean_text(text):
    """テキストから不要な文字を削除し、半角スペースに統一する関数"""
    text = re.sub(r'[・、,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_all_main_ingredients(main_ingrs_str, recognized_ingrs_set):
    """主要材料が認識された食材のセットにすべて含まれているかチェックするヘルパー関数"""
    if pd.isna(main_ingrs_str) or main_ingrs_str.strip() == "":
        return True
    cleaned_main_ingrs = clean_text(main_ingrs_str).split()
    return all(ingr in recognized_ingrs_set for ingr in cleaned_main_ingrs)

def recommend_recipes_based_on_main_and_required(recognized_ingredients, recipe_df, top_n=5):
    """
    認識された食材と主要材料、必要材料の共通点を考慮して関連度の高いレシピを提案する関数
    """
    required_cols_for_logic = ['recipe_name', 'required_ingredients', 'main_ingredients']
    for col in required_cols_for_logic:
        if col not in recipe_df.columns:
            raise ValueError(f"recipe_dfに'{col}'カラムが見つかりません。")

    recognized_ingredients_set = set(clean_text(" ".join(recognized_ingredients)).split())

    filtered_df = recipe_df[
        recipe_df['main_ingredients'].apply(
            lambda x: has_all_main_ingredients(x, recognized_ingredients_set)
        )
    ].copy()

    if filtered_df.empty:
        return pd.DataFrame(columns=list(recipe_df.columns) + ['similarity'])

    filtered_df['cleaned_required_ingredients'] = filtered_df['required_ingredients'].apply(clean_text)
    cleaned_recognized_ingredients_str = clean_text(" ".join(recognized_ingredients))

    all_ingredients_text_for_tfidf = filtered_df['cleaned_required_ingredients'].tolist() + [cleaned_recognized_ingredients_str]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_ingredients_text_for_tfidf)

    recipe_tfidf_matrix = tfidf_matrix[:-1]
    recognized_ingredients_tfidf = tfidf_matrix[-1]

    similarities = cosine_similarity(recognized_ingredients_tfidf, recipe_tfidf_matrix).flatten()

    filtered_df['similarity'] = similarities

    recommended_recipes_df = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_recipes_df[[col for col in recipe_df.columns if col != 'cleaned_required_ingredients'] + ['similarity']]


app = FastAPI()

# CORS設定: ウェブブラウザからのアクセスを許可するために重要
# このURLは、あなたのフロントエンド（Static Site）の公開URLです。
# Renderでデプロイするウェブアプリの実際のURLに合わせて変更してください。
# 例: "https://your-frontend-app.onrender.com"
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:5500",
    "null",
    "https://recipe-frontend-axty.onrender.com" # あなたのウェブアプリの公開URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSVファイルをロード
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'recipes.csv')
try:
    recipe_df_global = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    print("CSVファイルを正常に読み込みました。")
except Exception as e:
    print(f"CSVファイル読み込みエラー: {e}")
    recipe_df_global = pd.DataFrame()

# --- 画像認識APIエンドポイント ---
# ここにあなたの機械学習モデルの推論ロジックを記述します。
# 実際にはモデルのロードと推論ロジックを記述してください。
def run_ml_model_on_image(image_bytes: bytes) -> list[str]:
    """
    ここにあなたの機械学習モデルをロードし、画像から食材を推論するロジックを記述します。
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # ここで画像の前処理（リサイズ、正規化など）を行う
        # 例: image = image.resize((224, 224))
        # 例: image_array = np.array(image) / 255.0
        # 例: processed_image = np.expand_dims(image_array, axis=0) # モデルが期待する形式に変換

        # ここにモデルの推論コードを記述
        # recognized_ingredients_from_model = your_model.predict(processed_image)

        # 仮の認識結果を返す（実際のモデルに置き換えるまで）
        # アップロードされた画像サイズに応じて仮の食材リストを返す
        if image.width > 800 and image.height > 600:
            return ["chicken", "onion", "carrot"]
        elif image.width < 200 and image.height < 200:
            return ["egg", "tomato"]
        else:
            return ["potato", "cabbage"]

    except Exception as e:
        print(f"画像認識モデルの実行中にエラーが発生しました: {e}")
        return []

@app.post("/recognize_image")
async def recognize_image_api(file: UploadFile = File(...)):
    """
    画像をアップロードし、食材を認識してリストで返すAPIエンドポイント
    """
    try:
        image_bytes = await file.read()
        recognized_ingredients = run_ml_model_on_image(image_bytes)

        return {"recognized_ingredients": recognized_ingredients}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"画像認識中にエラーが発生しました: {e}")

# --- レシピ提案APIエンドポイント ---
@app.post("/recommend_recipes")
async def recommend_recipes_api(ingredients: list[str]):
    """
    認識された食材リストを受け取り、レシピを提案するAPIエンドポイント
    """
    if recipe_df_global.empty:
        raise HTTPException(status_code=500, detail="Recipe data not loaded.")
    
    recommendations = recommend_recipes_based_on_main_and_required(ingredients, recipe_df_global.copy())
    
    # NaN値をJSON対応の値に変換
    for col in ['main_ingredients', 'required_ingredients', 'instructions', 'dietary_restrictions']:
        if col in recommendations.columns:
            recommendations[col] = recommendations[col].replace({np.nan: ''})
    
    if 'similarity' in recommendations.columns:
        recommendations['similarity'] = recommendations['similarity'].replace({np.nan: None})

    return recommendations.to_dict(orient="records")

@app.get("/")
async def read_root():
    return {"message": "レシピ提案APIが稼働しています。/recommend_recipes にPOSTリクエストを送ってください。"}