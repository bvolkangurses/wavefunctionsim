
const express = require('express');
const http = require('http');
const path = require('path');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

// Serve everything in the 'public' folder as static files
app.use(express.static(path.join(__dirname, 'public')));

io.on('connection', (socket) => {
  console.log("A user connected:", socket.id);

  // If you want to do server-based computations or multi-user broadcasting,
  // you can listen for events from the client and respond here.
  // Example:
  // socket.on('updateWavefunction', (data) => {...});
  
  socket.on('disconnect', () => {
    console.log("A user disconnected:", socket.id);
  });
});

// Start the server
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log('Server listening on port', PORT);
});
