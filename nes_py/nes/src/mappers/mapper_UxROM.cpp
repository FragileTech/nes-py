//  Program:      nes-py
//  File:         mapper_UxROM.cpp
//  Description:  An implementation of the UxROM mapper
//
//  Copyright (c) 2019 Christian Kauten. All rights reserved.
//

#include <cstring>

#include "mappers/mapper_UxROM.hpp"
#include "log.hpp"

namespace NES {

MapperUxROM::MapperUxROM(Cartridge* cart) :
    Mapper(cart),
    has_character_ram(cart->getVROM().size() == 0),
    last_bank_pointer(cart->getROM().size() - 0x4000),
    select_prg(0) {
    if (has_character_ram) {
        character_ram.resize(0x2000);
        LOG(Info) << "Uses character RAM" << std::endl;
    }
}

NES_Byte MapperUxROM::readPRG(NES_Address address) {
    if (address < 0xc000)
        return cartridge->getROM()[((address - 0x8000) & 0x3fff) | (select_prg << 14)];
    else
        return cartridge->getROM()[last_bank_pointer + (address & 0x3fff)];
}

NES_Byte MapperUxROM::readCHR(NES_Address address) {
    if (has_character_ram)
        return character_ram[address];
    else
        return cartridge->getVROM()[address];
}

void MapperUxROM::writeCHR(NES_Address address, NES_Byte value) {
    if (has_character_ram)
        character_ram[address] = value;
    else
        LOG(Info) <<
            "Read-only CHR memory write attempt at " <<
            std::hex <<
            address <<
            std::endl;
}

void MapperUxROM::dump_state(char *buffer) {
    *reinterpret_cast<size_t*>(buffer) = character_ram.size();
    buffer += sizeof(size_t);
    memcpy(buffer, character_ram.data(), character_ram.size());
}

void MapperUxROM::load_state(const char *buffer) {
    character_ram.resize(*reinterpret_cast<const size_t*>(buffer));
    buffer += sizeof(size_t);
    memcpy(character_ram.data(), buffer, character_ram.size());
}

}  // namespace NES
