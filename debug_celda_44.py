# Diagnóstico para el error de plotting en celda 44
import matplotlib.pyplot as plt
import numpy as np

def diagnose_training_history(history):
    """
    Función de diagnóstico para identificar el problema con el historial de entrenamiento
    """
    print("🔍 DIAGNÓSTICO DEL HISTORIAL DE ENTRENAMIENTO")
    print("=" * 50)
    
    # Verificar si history existe y no está vacío
    if not history:
        print("❌ ERROR: 'history' está vacío o es None")
        return False
    
    print(f"✅ History existe y tiene tipo: {type(history)}")
    
    # Mostrar todas las claves disponibles
    print(f"\n📋 CLAVES DISPONIBLES EN HISTORY:")
    for key in history.keys():
        value = history[key]
        if isinstance(value, list):
            print(f"   {key}: lista con {len(value)} elementos")
            if len(value) > 0:
                print(f"      Primeros valores: {value[:3]}")
            else:
                print(f"      ⚠️  LISTA VACÍA")
        else:
            print(f"   {key}: {type(value)} = {value}")
    
    # Verificar claves específicas que necesita la función de plotting
    required_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    missing_keys = []
    empty_keys = []
    
    print(f"\n🎯 VERIFICANDO CLAVES REQUERIDAS:")
    for key in required_keys:
        if key not in history:
            missing_keys.append(key)
            print(f"   ❌ {key}: NO EXISTE")
        elif not history[key] or len(history[key]) == 0:
            empty_keys.append(key)
            print(f"   ⚠️  {key}: EXISTE PERO ESTÁ VACÍO")
        else:
            print(f"   ✅ {key}: OK ({len(history[key])} elementos)")
    
    # Buscar claves similares que podrían ser las correctas
    print(f"\n🔍 BUSCANDO CLAVES SIMILARES:")
    all_keys = list(history.keys())
    similar_patterns = {
        'train_acc': ['train_accuracy', 'training_acc', 'training_accuracy', 'acc_train'],
        'val_acc': ['val_accuracy', 'validation_acc', 'validation_accuracy', 'acc_val'],
        'train_loss': ['training_loss', 'loss_train'],
        'val_loss': ['validation_loss', 'loss_val']
    }
    
    for expected_key, patterns in similar_patterns.items():
        if expected_key in missing_keys or expected_key in empty_keys:
            found_alternatives = []
            for pattern in patterns:
                matching_keys = [k for k in all_keys if pattern in k.lower()]
                found_alternatives.extend(matching_keys)
            
            if found_alternatives:
                print(f"   💡 Para '{expected_key}' encontré: {found_alternatives}")
            else:
                print(f"   ❌ No encontré alternativas para '{expected_key}'")
    
    # Diagnóstico específico del error
    print(f"\n🚨 DIAGNÓSTICO DEL ERROR:")
    if 'train_acc' not in history:
        print("   CAUSA PRINCIPAL: 'train_acc' no existe en history")
        print("   SOLUCIÓN: El training loop no está guardando accuracy")
    elif len(history['train_acc']) == 0:
        print("   CAUSA PRINCIPAL: 'train_acc' existe pero está vacío")
        print("   SOLUCIÓN: El training loop no está calculando accuracy correctamente")
    else:
        print("   CAUSA DESCONOCIDA: train_acc parece estar bien")
    
    return len(missing_keys) == 0 and len(empty_keys) == 0

def create_fixed_plotting_function():
    """
    Crear función de plotting que maneja datos faltantes
    """
    
    def plot_advanced_training_history_fixed(history):
        """
        Versión corregida que maneja datos faltantes graciosamente
        """
        print("📊 PLOTTING CON MANEJO DE ERRORES")
        print("=" * 40)
        
        # Verificar datos primero
        if not diagnose_training_history(history):
            print("❌ No se puede hacer el plot debido a datos faltantes")
            return
        
        # Intentar mapear claves alternativas
        key_mapping = {
            'train_loss': ['train_loss', 'training_loss', 'loss_train'],
            'val_loss': ['val_loss', 'validation_loss', 'loss_val'],
            'train_acc': ['train_acc', 'train_accuracy', 'training_acc', 'training_accuracy'],
            'val_acc': ['val_acc', 'val_accuracy', 'validation_acc', 'validation_accuracy']
        }
        
        mapped_data = {}
        for standard_key, possible_keys in key_mapping.items():
            found = False
            for possible_key in possible_keys:
                if possible_key in history and history[possible_key]:
                    mapped_data[standard_key] = history[possible_key]
                    print(f"✅ Mapeando '{possible_key}' -> '{standard_key}'")
                    found = True
                    break
            
            if not found:
                print(f"❌ No se encontraron datos para '{standard_key}'")
                return
        
        # Crear el plot
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(mapped_data['train_loss']) + 1)
        
        # Plot 1: Loss
        ax1.plot(epochs, mapped_data['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, mapped_data['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Durante Entrenamiento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        ax2.plot(epochs, mapped_data['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, mapped_data['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Durante Entrenamiento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Mostrar estadísticas
        best_epoch = np.argmax(mapped_data['val_acc']) + 1
        print(f"\n📊 RESUMEN:")
        print(f"   🏆 Mejor época: {best_epoch}")
        print(f"   📈 Mejor Val Accuracy: {max(mapped_data['val_acc']):.4f}")
        print(f"   📉 Loss final: Train {mapped_data['train_loss'][-1]:.4f}, Val {mapped_data['val_loss'][-1]:.4f}")
    
    return plot_advanced_training_history_fixed

# Crear la función corregida
plot_advanced_training_history_fixed = create_fixed_plotting_function()

print("✅ Funciones de diagnóstico creadas")
print("\n💡 INSTRUCCIONES DE USO:")
print("1. diagnose_training_history(training_history)")
print("2. plot_advanced_training_history_fixed(training_history)")