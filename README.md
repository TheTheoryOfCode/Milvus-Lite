***

### Milvus Lite — A Vector Database for rapid prototyping 

![](https://cdn-images-1.medium.com/max/1600/1*t9FhXuvDCCX04L6ZUqINkg.png)

Milvus lite — Open Source Vector Database

***

#### The use of Vector Databases

Vector databases are known to be used widely within A.I. enabled applications, but actual usage transcendent to many other use cases and general purpose applications. 

*One of the most prominent and generalized use of vector databases are to find ****similarity searches or nearest neighbours. ***

> However, vector database are not the only way to find the similarity or nearest neighbours, machine learning libraries like KNN or many other data structures can also be used for the same. 

And most often than not, we may need some kind of prototype to check the applicability and validity and we need to do it fast. 

> A simple prototype can be written for any vector database, However, I’ve found Milvus Lite to be working for me on my machine easily and without any problems.

***

#### Milvus Lite — Vector Database 

![](https://cdn-images-1.medium.com/max/1600/1*LduQcvFAIs2iF24Xq2NM3A.jpeg)

Vector Databases

[***Milvus Lite***](https://github.com/milvus-io/milvus-lite) is an open-source, lightweight version of the Milvus vector database for rapid prototyping. It's possible that you may end us using `Milvus` itself, but if you want to take the first step with a vector database and understanding how it works, you can use ***Milvus Lite***

***

#### The Use Case

Let’s assume that we’re evaluating a use case where we have details of the books in the following form

*   Book Name
*   Author Name
*   Excerpt / Selected Texts from the book

We want to build a system where I should be able to get the ***name and author of the book*** by sharing the ***partial excerpts or partial texts***. 

Basically the question will look something like this. 

Tell be the book and author of the book which contains the following texts

> ***“nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.”***

The answer should be 

> Book Name — The Hobbit

> Author Name — J.R.R. Tolkien

***

#### Implementing Milvus Lite Vector Database — Using Python

First we need to have a sample data of books, author and the excerpts. For simplicity purposes, I’m using a limited number of records, 7 to be precise. 

    book_data = [
        {
            "title": "Moby Dick",
            "author": "Herman Melville",
            "excerpt": "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation."
        },
        {
            "title": "Dune",
            "author": "Frank Herbert",
            "excerpt": "A beginning is the time for taking the most delicate care that the balances are correct. To begin your study of the life of Muad'Dib, then, take care that you first place him in his time: born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most special care that you locate him in his place: the planet Arrakis, known as Dune."
        },
        {
            "title": "Nineteen Eighty-Four",
            "author": "George Orwell",
            "excerpt": "It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him."
        },
        {
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "excerpt": "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort."
        },
        {
            "title": "Meditations",
            "author": "Marcus Aurelius",
            "excerpt": "You have power over your mind - not outside events. Realize this, and you will find strength. The happiness of your life depends upon the quality of your thoughts: therefore, guard accordingly, and take care that you entertain no notions unsuitable to virtue and reasonable nature."
        },
        {
            "title": "A Study in Scarlet",
            "author": "Arthur Conan Doyle",
            "excerpt": "From a drop of water, a logician could infer the possibility of an Atlantic or a Niagara without having seen or heard of one or the other. So all life is a great chain, the nature of which is known whenever we are shown a single link of it. This is the science of deduction and analysis."
        },
        {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "excerpt": "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered the rightful property of some one or other of their daughters."
        }
    ]

Once I have the data, l’ll create `pandas dataframe` out of it

    import pandas as pd

    df = pd.DataFrame(book_data)

***

#### Vectorization of the Data

Vector databases can only store floating point values and all texts must be converted into floating point numbers of a specific dimension so that it can be store in a vector database.

One of the most preferred way to do the same is to use the `sentence-transformers` and we can use the same with one of the light weight model called ***all-MiniLM-L6-v2***

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    DIMENSION = 384

This model generates vector of dimension ***384 ***which means it will generate ***384*** unique floating point values to represent the text. For smaller texts, like what we’re using in our example, the dimension of ***384*** is good enough, for larger texts, higher dimensions are desirable 

***

#### Using the Milvus Lite Database

To use the same, first we need to import the same in our python code as

    from pymilvus import MilvusClient, DataType

Now we need to create a database (i.e. vector database) which will be used to store the vector in the local disk. Let’s call the database as `books.db`

    vec_db_client = MilvusClient(uri='books.db')

Just like normal SQL databases, we can create a schema and fields (a.k.a columns). One of the field must be the primary key. 

    vec_db_schema = MilvusClient.create_schema(
            auto_id=False,   # no auto generate the primary key
            enable_dynamic_field=False, # allow data which are not officially part of the defined fields
        )

    # create the primary key
    # since book title is unique, it can be used as a primary key

    vec_db_schema.add_field(
            field_name="title", 
            datatype=DataType.VARCHAR,  
            max_length=255, 
            is_primary=True, 
            auto_id=False)

    # create other fields of the schema
    vec_db_schema.add_field(
          field_name="author", 
          datatype=DataType.VARCHAR, 
          max_length=255)

    vec_db_schema.add_field(
              field_name="excerpt", 
              datatype=DataType.VARCHAR, 
              max_length=1024)

Now, we have created a schema to accommodate book title, author and excerpt. Now we need to create another field which will hold the vector embeddings of the excerpts and while doing that we need to specifythe vector dimension

    vec_db_schema.add_field(
          field_name="vector_emb",  
          datatype=DataType.FLOAT_VECTOR, 
          dim=DIMENSION
    )

Now, we need to create the vector embedding for all the data in the data frame and this can be done in one line by using the data frame `.apply` method.

    df['vector_emb'] = df['excerpt'].apply(
        lambda x: model.encode(x, 
        show_progress_bar=True)
    )

This will create a new column in the pandas data frame, corresponding to the excerpt in the same row.

***

#### Storing of the data frame in vector DB

To store the data frame in the created vector database, we need to tell the vector database about how to create and manage the indexes. Here is how we do that

    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="vector_emb",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 16}
    )

In the function `add_index` , we’re provide the following 

*   ***field\_name ***— The actual field in the data frame which contains the vector embeddings
*   ***index\_type*** — `IVF_FLAT` means that the vector indexes should be stored in an inverted file format, which means clustering the vectors into manageable regions using ***K — Mean Clustering***
*   ***metric\_type ***— `COSINE` specifies the algorithm that will be used to measure the distance between vectors. `COSINE`means it will use cosine similarity, while `L2` means euclidian distance. There are other options like `JACCARD` and `HAMMING` 
*   ***params ***— `"nlist" : 16"` , this defines the number of clusters the vector space is divided into. This is done to increase the speed of search. you can consider this as a top level index. 

#### Creation of Database Collection

The vector database created earlier i.e.`books.db` doesn’t stores the data directly. We need to create a collection on top of the database and then insert the data into the collection. 

> If you’ve worked with MongoDB earlier, this should sound familiar to you

    # creating a collection

    vec_db_client.create_collection(
        collection_name="books",   # Collection Name
        schema=vec_db_schema,      # Using the Schema
        index_params=index_params  # Using Index Parameter
    )

    # inserting the data into the collection
    vec_db_client.insert(
        collection_name="books",      # collection name
        data=df.to_dict('records')    # data to insert in the collection
    )

Once the above commands are successfully executed, we have a vector database which has stored the vector embeddings. Now, we can search the database

***

#### Searching the Vector Database

Vector databases doesn’t allow text based searches, we need to vectorize the search text just as we did while storing the data.

We’ll convert the text into the corresponding vector before searching using the same model 

Here is how we can do the same.

    # search the text 
    search_which_book = "bright cold day in April, 
                and the clocks were striking thirteen."


    # create a query vectir
    query_vector = model.encode(
        search_which_book, 
        show_progress_bar=True
    )

once we have the query vector we can use the same for searching as

    #search withing the vector database

    search_result = vec_db_client.search(
        collection_name="books",          # collection name
        data=[query_vector],              # query vector
        anns_field="vector_emb",          # field whcih contains the vector embedding
        limit=2,                          # number of results
        output_fields=["title", "author"] # output fields
    )

The output can be decoded as

    search_result[0][0]['entity'].      # The first result
    search_result[0][0]['distance']     # Cosine Distance

    search_result[0][1]['entity'].      # The second result
    search_result[0][1]['distance']     # Cosine Distance

This is how we can quickly create a vector database and use the same for similarity search. Now the vector database created on the disk as a result of the code written above can be used anywhere for querying purposes.

***

Thanks for Reading..

Daksh
