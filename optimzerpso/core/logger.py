import logging

class Logger:
    # Colores ANSI
    RESET         = "\033[0m"
    RED           = "\033[31m"
    GREEN         = "\033[32m"
    YELLOW        = "\033[33m"
    BLUE          = "\033[34m"
    PURPLE        = "\033[35m"
    CYAN          = "\033[36m"
    WHITE         = "\033[37m"
    BRIGHT_BLACK  = "\033[90m"
    BRIGHT_RED    = "\033[91m"
    BRIGHT_GREEN  = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE   = "\033[94m"
    BRIGHT_PURPLE = "\033[95m"
    BRIGHT_CYAN   = "\033[96m"
    BRIGHT_WHITE  = "\033[97m"
    
    _instance = None
    
    def __new__(cls, level=logging.DEBUG):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, level=logging.DEBUG):
        if self._initialized:
            return
            
        # Creamos un logger con un nombre específico
        self.logger = logging.getLogger('optimzerpso')
        self.logger.setLevel(level)
        
        # Removemos handlers existentes para evitar duplicados
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Creamos un handler que se encarga de imprimir en consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Formato de los mensajes
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Añadimos el handler al logger
        self.logger.addHandler(console_handler)
        
        # Evitamos la propagación a loggers padre
        self.logger.propagate = False
        
        self._initialized = True

    def alerta(self, mensaje):
        self.logger.warning(f"{self.YELLOW}⚠️ ALERTA: {mensaje}{self.RESET}")

    def info(self, mensaje):
        self.logger.info(f"{self.CYAN}🐞 INFO: {mensaje}{self.RESET}")

    def error(self, mensaje):
        self.logger.error(f"{self.RED}❌ ERROR: {mensaje}{self.RESET}")

    def exito(self, mensaje):
        self.logger.info(f"{self.GREEN}✅ ÉXITO: {mensaje}{self.RESET}")
    
    def canvas(self, mensaje):
        self.logger.info(f"{self.PURPLE}🎨 CANVAS: {mensaje}{self.RESET}")

    def worker(self, mensaje):
        self.logger.info(f"{self.BRIGHT_GREEN}👷 WORKER: {mensaje}{self.RESET}")

    def fire(self, mensaje):
        self.logger.info(f"{self.BRIGHT_RED}🔥 FIRE: {mensaje}{self.RESET}")

logger = Logger()