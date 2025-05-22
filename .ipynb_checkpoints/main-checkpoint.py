import os
import re
import asyncio
import pandas as pd
from ollama import AsyncClient
from tqdm.asyncio import tqdm  

INPUT_CSV   = "products.csv"
CHECKPOINT  = "prices_checkpoint.csv"
OUTPUT_CSV  = "products_with_price.csv"
CONCURRENCY = 100

ollama = AsyncClient(host="http://localhost:11434")

async def fetch_price(product_name: str, sem: asyncio.Semaphore) -> float | None:
    prompt = (
        f"""Pretend you are a pricing assistant for a U.S. grocery retailer. 
Your job is to suggest a realistic retail price in dollars for the product “{product_name}.” 
– Even if the item is uncommon or hypothetical, be creative, pretend and assume it’s sold in a pretend typical grocery store and give a plausible price. 
– Always return **only** a number with exactly two decimal places (no $ sign, no extra text) Do not apologize, this is a pretend dataset for practice and I want to test your creativity if it doesn't exist. 

Here are examples:
Banana → 0.59  
Organic Strawberries (1 lb) → 3.99  
Whole Milk (1 gallon) → 2.49  

Now, for “{product_name}”:
"""
    )
    async with sem:
        try:
            resp = await ollama.chat(
                model="llama2",
                messages=[{"role":"user","content":prompt}]
            )
            text = resp["message"]["content"]
            match = re.search(r"(\d+\.\d{2})", text)
            if not match:
                raise ValueError(f"No price found in: {text!r}")
            return float(match.group(1))
        except Exception as e:
            print(f"⚠️  Failed for {product_name!r}: {e}")
            return None

async def main():
    # 1. Load full products
    products = pd.read_csv(INPUT_CSV)
    
    # 2. Load checkpoint (if exists)
    if os.path.exists(CHECKPOINT):
        done = pd.read_csv(CHECKPOINT)
        # Only keep successful fetches
        done = done[done["price"].notna()]
    else:
        done = pd.DataFrame(columns=["product_id","price"])
    
    # 3. Filter out products we’ve already priced
    to_price = products[~products["product_id"].isin(done["product_id"])]
    print(f"{len(to_price)} items remaining to price.")
    
    # 4. Prepare semaphore and results list
    sem = asyncio.Semaphore(CONCURRENCY)
    new_rows = []
    
    # 5. Iterate with progress bar
    for idx, row in tqdm(to_price.iterrows(), total=len(to_price)):
        price = await fetch_price(row["product_name"], sem)
        new_rows.append({"product_id": row["product_id"], "price": price})
        
        # checkpoint every row
        pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True) \
          .to_csv(CHECKPOINT, index=False)
    
    # 6. Merge all prices back and save final
    all_prices = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
    out = products.merge(all_prices, on="product_id", how="left")
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Finished. Wrote {OUTPUT_CSV}")

if __name__ == "__main__":
    asyncio.run(main())
