from pymilvus import model


ef = model.DefaultEmbeddingFunction()

docs = [
    "刘禹锡 772年-842年 字梦得 荥阳（今河南省郑州市荥阳市）人，祖籍洛阳（今河南省洛阳市.",
    "刘禹锡诗文俱佳，涉猎题材广泛，与白居易并称“刘白”，与柳宗元并称“刘柳”，与韦应物、白居易合称“三杰”，留有《陋室铭》《竹枝词》《杨柳枝词》《乌衣巷》等名篇。",
    "刘禹锡的山水诗，改变了大历、贞元诗人襟幅狭小、气象萧瑟的风格，而常常是写一种超出空间实距的、半虚半实的开阔景象",
]

embeddings = ef.encode_documents(docs)
print("Embeddings:", embeddings, type(embeddings), len(embeddings), embeddings[0].shape)