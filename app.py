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

#CLIENT_PORT = os.environ["CLIENT_PORT"]

class OdooConfig:
    uid = None
    odoo_url = os.getenv("odoo_url")
    db_name = os.getenv("odoo_db_name")
    username = os.getenv("odoo_username")
    password = os.getenv("odoo_password")
    models_endpoint = f"{odoo_url}/xmlrpc/2/object"
    get_models_proxy = lambda self: xmlrpc.client.ServerProxy(self.models_endpoint)

odoo = OdooConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    common_endpoint = f"{odoo.odoo_url}/xmlrpc/2/common"
    common_proxy = xmlrpc.client.ServerProxy(common_endpoint)
    odoo.uid = common_proxy.authenticate(odoo.db_name, odoo.username, odoo.password, {})
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

#api

@app.get('/')
def main():
    return {'Message' : 'Welcome to api'}

@app.get('/{name}')
def hello(name : str):
    return {'message' : f'Welcome to api {name }'}

@app.post('/search')
async def search(query: str):
    try:
        search_text = query
        result_ids = Search(search_text, 30, index)

        ids = [
            int(id)
            for id in result_ids
        ]

        model_name = "product.template"
        domain = [("id", "in", ids)]
        fields = ["name", "list_price", "standard_price", "qty_available", "description_sale", "description_purchase", "description"]
        models = odoo.get_models_proxy()
        products = models.execute_kw(odoo.db_name, odoo.uid, odoo.password, model_name, "search_read", [domain], {"fields": fields})

        return products
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    