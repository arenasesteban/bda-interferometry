# bda-interferometry

## Instalación

### Prerequisitos

- **Python**: 3.10 o superior.
- **Micromamba**: Gestor de entornos virtuales y dependencias.

### Pasos de instalación

1. Clonar repositorio:
```bash
git clone https://github.com/arenasesteban/bda-interferometry.git
cd bda-interferometry
```

2. Crear un entorno virtual:
```bash
micromamba create -n bda-env
micromamba activate bda-env
```

3. Instalar Pyralysis:
```bash
cd ..
git clone https://gitlab.com/clirai/pyralysis.git
cd pyralysis
pip install --extra-index-url https://artefact.skao.int/repository/pypi-internal/simple -e .
```

## Estándar de Commits

```bash
# [ACTION]: [message]
git commit -m "ADD: script to generate simulated dataset."
```

- **ACTION**: Una de las siguientes acciones, en mayúsculas.
- **message**: Una breve descripción del cambio realizado.

Este proyecto sigue una convención simple para los mensajes de commit, diseñada para mantener claridad y consistencia en el historial de cambios.

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