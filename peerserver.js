/**
 * Local PeerJS Signaling Server
 * Replaces the unreliable default peerjs.com public server.
 * Run: node peerserver.js   (port 9000)
 */
const { PeerServer } = require('peer');

const peerServer = PeerServer({
    port: 9000,
    path: '/peerjs',
    allow_discovery: true,
});

peerServer.on('connection', (client) => {
    console.log(`[PeerServer] Client connected: ${client.getId()}`);
});

peerServer.on('disconnect', (client) => {
    console.log(`[PeerServer] Client disconnected: ${client.getId()}`);
});

console.log('✅ PeerJS signaling server running on http://localhost:9000/peerjs');
