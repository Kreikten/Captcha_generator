import uvicorn

uvicorn.run(

    "Captcha_generator.app:application",
    reload=True,
    port=80
    #host="127.0.0.1"


)

