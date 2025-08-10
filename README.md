# bda-interferometry

## Estándar de Commits

### 1. Descripción
Este proyecto sigue una convención simple para los mensajes de commit, diseñada para mantener claridad y consistencia en el historial de cambios.

---

### 2. Estructura de un commit

```bash
# ACTION: message
ADD: script to generate simulated dataset.
```

- **ACTION** → Palabra clave en mayúsculas que describe el tipo de cambio.
- **message** → Descripción breve y clara en inglés, en tiempo presente.

---

### 3. Tabla de acciones

| Acción  | Uso recomendado |
|---------|-----------------|
| **ADD** | Añadir código, archivos, funciones, endpoints, tests o cualquier recurso nuevo. |
| **CHG** | Cambiar o mejorar código existente que no soluciona un bug. |
| **FIX** | Corregir errores, bugs o fallos detectados. |
| **MERGE** | Integrar cambios de otra rama o resolver conflictos de merge. |
| **DEL** | Eliminar código, archivos o funcionalidades que ya no se usan. |
| **DOC** | Cambios exclusivamente en documentación. |
| **TEST** | Cambios exclusivamente en pruebas unitarias o de integración. |
| **CONF** | Cambios en archivos de configuración, scripts de despliegue o entornos. |