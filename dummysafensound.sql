-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Oct 17, 2025 at 01:24 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `safensound`
--

-- --------------------------------------------------------

--
-- Table structure for table `history`
--

CREATE TABLE `history` (
  `history_id` int(11) NOT NULL,
  `action` enum('Alert Acknowledged','Emergency Detected') NOT NULL,
  `date` date DEFAULT NULL,
  `time` time DEFAULT NULL,
  `room_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `history`
--

INSERT INTO `history` (`history_id`, `action`, `date`, `time`, `room_id`) VALUES
(1, 'Emergency Detected', '2025-10-01', '08:15:30', 1),
(2, 'Alert Acknowledged', '2025-10-01', '08:16:45', 1),
(3, 'Emergency Detected', '2025-10-03', '14:22:10', 1),
(4, 'Alert Acknowledged', '2025-10-03', '14:23:00', 1),
(5, 'Emergency Detected', '2025-10-05', '09:45:20', 1),
(6, 'Alert Acknowledged', '2025-10-05', '09:46:15', 1),
(7, 'Emergency Detected', '2025-10-08', '16:30:55', 1),
(8, 'Alert Acknowledged', '2025-10-08', '16:32:10', 1),
(9, 'Emergency Detected', '2025-10-12', '11:20:40', 1),
(10, 'Alert Acknowledged', '2025-10-12', '11:21:30', 1),
(11, 'Emergency Detected', '2025-10-02', '10:05:15', 2),
(12, 'Alert Acknowledged', '2025-10-02', '10:06:20', 2),
(13, 'Emergency Detected', '2025-10-04', '13:40:30', 2),
(14, 'Alert Acknowledged', '2025-10-04', '13:41:45', 2),
(15, 'Emergency Detected', '2025-10-07', '15:15:50', 2),
(16, 'Alert Acknowledged', '2025-10-07', '15:17:05', 2),
(17, 'Emergency Detected', '2025-10-10', '08:55:25', 2),
(18, 'Alert Acknowledged', '2025-10-10', '08:56:40', 2),
(19, 'Emergency Detected', '2025-10-14', '17:10:35', 2),
(20, 'Alert Acknowledged', '2025-10-14', '17:11:50', 2),
(21, 'Emergency Detected', '2025-10-01', '12:30:45', 3),
(22, 'Alert Acknowledged', '2025-10-01', '12:31:55', 3),
(23, 'Emergency Detected', '2025-10-06', '09:20:10', 3),
(24, 'Alert Acknowledged', '2025-10-06', '09:21:25', 3),
(25, 'Emergency Detected', '2025-10-09', '14:45:30', 3),
(26, 'Alert Acknowledged', '2025-10-09', '14:46:40', 3),
(27, 'Emergency Detected', '2025-10-11', '11:05:20', 3),
(28, 'Alert Acknowledged', '2025-10-11', '11:06:35', 3),
(29, 'Emergency Detected', '2025-10-15', '16:25:50', 3),
(30, 'Alert Acknowledged', '2025-10-15', '16:27:05', 3),
(31, 'Emergency Detected', '2025-10-16', '07:40:15', 1),
(32, 'Emergency Detected', '2025-10-16', '08:20:30', 2),
(33, 'Alert Acknowledged', '2025-10-16', '08:21:45', 2),
(34, 'Emergency Detected', '2025-10-17', '10:15:20', 3),
(35, 'Alert Acknowledged', '2025-10-17', '10:16:35', 3);

-- --------------------------------------------------------

--
-- Table structure for table `room`
--

CREATE TABLE `room` (
  `room_id` int(11) NOT NULL,
  `room_name` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `room`
--

INSERT INTO `room` (`room_id`, `room_name`) VALUES
(1, 'Room 1'),
(2, 'Room 2'),
(3, 'Room 3');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `history`
--
ALTER TABLE `history`
  ADD PRIMARY KEY (`history_id`),
  ADD KEY `fk_room` (`room_id`);

--
-- Indexes for table `room`
--
ALTER TABLE `room`
  ADD PRIMARY KEY (`room_id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `history`
--
ALTER TABLE `history`
  MODIFY `history_id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=36;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `history`
--
ALTER TABLE `history`
  ADD CONSTRAINT `fk_room` FOREIGN KEY (`room_id`) REFERENCES `room` (`room_id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
