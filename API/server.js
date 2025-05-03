const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const app = express();
const PORT = 2000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.post('/prediccion', (req, res) => {
  const input = JSON.stringify(req.body);

  const py = spawn('python', [path.join(__dirname, 'modelo', 'predict.py')]);

  let resultado = '';
  py.stdout.on('data', (data) => {
    resultado += data.toString();
  });

  py.stderr.on('data', (data) => {
    console.error('Error en Python:', data.toString());
  });

  py.on('close', () => {
    try {
      const json = JSON.parse(resultado);
      res.json(json);
    } catch (error) {
      console.error('❌ Error al interpretar la respuesta:', error);
      res.status(500).json({ error: 'Respuesta inválida del script Python' });
    }
  });

  py.stdin.write(input);
  py.stdin.end();
});

app.listen(PORT, () => {
  console.log(`Servidor iniciado en http://localhost:${PORT}`);
});
