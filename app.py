from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import logging
import logging.config
from utils.load_resourses import load_image_embedding
from src.methods.search import Search
import xmlrpc.client
from contextlib import asynccontextmanager
import os
from googletrans import Translator
from dotenv import load_dotenv
import copy
from typing import Optional
import re


load_dotenv()
    

class OdooConfig:
    uid = None
    odoo_url = os.environ.get("odoo_url")
    db_name = os.environ.get("odoo_db_name")
    username = os.environ.get("odoo_username")
    password = os.environ.get("odoo_password")
    models_endpoint = f"{odoo_url}/xmlrpc/2/object"
    common_endpoint = f"{odoo_url}/xmlrpc/2/common"
    get_models_proxy = lambda self: xmlrpc.client.ServerProxy(self.models_endpoint)
    get_common_proxy = lambda self: xmlrpc.client.ServerProxy(self.common_endpoint)

odoo = OdooConfig()

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    odoo.uid = odoo.get_common_proxy().authenticate(odoo.db_name, odoo.username, odoo.password, {})
    print("Successfully connected to Odoo!")
    yield 
    print("Disconnecting from Odoo...")
    odoo.uid = None
    print("Disconnected from Odoo!")

    
app = FastAPI(lifespan=lifespan)



app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

#CORS
origin = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class request_body(BaseModel):
    query : str

class SearchResult(BaseModel):
    result_paths: list

index = load_image_embedding()

app_logger = logging.getLogger("app_logger")

@app.get('/search')
async def search(query: Optional[str] = None):
    try:
        model_name = "product.template"
        fields = ["name", "list_price", "standard_price", "qty_available", "description_sale", "description_purchase", "description", "image_256"]
        models = odoo.get_models_proxy()

        all_products = []
        # filter published products

        if query is None:
            # limit 100 products
            all_products = models.execute_kw(odoo.db_name, odoo.uid, odoo.password, model_name, 'search_read', [[['website_published', '=', True]]], {'fields': fields, 'limit': 100})
            return {
                "raw_text_products": all_products,
            }
        else:
            all_products = models.execute_kw(odoo.db_name, odoo.uid, odoo.password, model_name, "search_read", [[['website_published', '=', True]]], {'fields': fields})
        
            search_text = query
            english_search_text = await translate(search_text)

            result_ids = Search(english_search_text, 10, index)

            
            
            raw_text_products = []
            ai_products = []

            search_text_words = split_sentence(search_text)

            for res_id in result_ids:
                for product in all_products:
                    if product["id"] == res_id.get("id"):
                        ai_product = copy.deepcopy(product)
                        ai_product["distance"] = res_id.get("distance")
                        ai_products.append(ai_product)
                        break

            seen_products = set()


            for word in search_text_words:
                for product in all_products:
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    if pattern.search(product["name"]):
                        product["name"] = pattern.sub(f"<b>{word}</b>", product["name"]).capitalize()
                        product_tuple = tuple(product.items())
                        if product_tuple not in seen_products:
                            raw_text_products.append(product)
                            seen_products.add(product_tuple)



            return {
                "ai_products": ai_products,
                "raw_text_products": raw_text_products
            }
    except Exception as e:
        app_logger.error(e)
        return {"message": "Something went wrong!"}
    
@app.post('/translate')
async def translate(query: str):
    try:
        translator = Translator(service_urls=['translate.google.com'])
        translation = translator.translate(query, dest='en')
        return translation.text
    except Exception as e:
        app_logger.error(e)
        return {"message": "Something went wrong!"}
    

@app.get('/split-text')
def split_sentence(input_sentence):
    words = input_sentence.split()
    result = []

    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            # check if the substring is a number then skip
            if words[i].isdigit():
                continue
            result.append(" ".join(words[i:j]))

    result.sort(key=lambda x: len(x), reverse=True)

    return result

    